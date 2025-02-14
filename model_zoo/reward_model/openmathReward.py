import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from tensordict import TensorDict

from verl import DataProto
from verl.workers.reward_model.megatron import MegatronRewardModel


class OpenmathRewardModel(MegatronRewardModel):
    @torch.no_grad()
    def compute_reward(self, data: DataProto,
                       format_reward: int = 1,
                       answer_reward: float = 0.0) -> DataProto:
        if self.config.param_offload:
            self.load_params_to_cuda()

        if self.use_different_tokenizer:
            data, ori_values = self.re_encode_by_rm_tokenizer(data)

        input_ids = data.batch['input_ids']  # (bs, seq_len')
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']

        responses = data.batch['responses']
        batch_size = responses.size(0)
        response_length = responses.size(1)

        solution_str = self.sft_tokenizer.decode(input_ids)

        from verl.utils.reward_score.kk import extract_solution, validate_response_structure
        equation, extract_solution_str = extract_solution(solution_str=solution_str)  # 提取<answer>*</answer>间的内容
        format_correct = validate_response_structure(extract_solution_str)
        format_score = format_reward if format_correct else -abs(format_reward)

        with torch.no_grad():
            output = self.forward_batch(data)
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                logits = torch.cat([o['logits'] for o in output], dim=0)
            else:
                logits = torch.empty(
                    (input_ids.shape[0], input_ids.shape[1]),
                    dtype=torch.bfloat16,  # TODO(sgm): check why is bfloat16
                    device=input_ids.device)
            # broadcast across pp ranks
            torch.distributed.broadcast(tensor=logits,
                                        src=mpu.get_pipeline_model_parallel_last_rank(),
                                        group=mpu.get_pipeline_model_parallel_group(),
                                        async_op=False)

        # 确定每个样本的最后一个有效token位置
        ends = attention_mask.cumsum(dim=-1).argmax(dim=-1)  # (bs,)

        # 提取最后一个token的logits (batch_size, vocab_size)
        last_token_logits = logits[torch.arange(batch_size), ends]

        # 获取True和False的logits (batch_size, 2)
        true_false_logits = last_token_logits[:, [14004, 8996]]

        if self.use_different_tokenizer:
            data.batch.update(ori_values)
            input_ids = ori_values['input_ids']
            attention_mask = ori_values['attention_mask']
            position_ids = ori_values['position_ids']

        # 确保后续处理兼容（如有需要）
        token_level_rewards = torch.softmax(true_false_logits, dim=-1)

        if self.config.param_offload:
            self.offload_params_to_cpu()
        else:
            torch.cuda.empty_cache()

        # 将结果存入batch
        batch = TensorDict({
            'rm_scores': token_level_rewards,  # 现在包含True/False的logits
            'true_false_logits': token_level_rewards  # 可选，明确字段
        }, batch_size=input_ids.shape[0])

        return DataProto(batch=batch)