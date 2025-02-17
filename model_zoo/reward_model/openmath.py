import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from tensordict import TensorDict

from verl import DataProto
from verl.workers.fsdp_workers import RewardModelWorker


class CustomRewardModelWorker(RewardModelWorker):
    def __init__(self, config):
        super().__init__()
        assert self.use_remove_padding == True, "use_remove_padding should be false for custom reward model"
        assert  self.ulysses_sequence_parallel_size == 1, "ulysses_sequence_parallel_size does not support custom reward model"

        from transformers import AutoModelForCausalLM
        self.load_method = AutoModelForCausalLM

    def _switch_chat_template(self, data: DataProto):
        # TODO: custom template
        return super()._switch_chat_template(data)

    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis, rearrange

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            input_ids_rmpad, indices, cu_seqlens, _ = unpad_input(input_ids.unsqueeze(-1),
                                                       attention_mask)  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                  indices).transpose(0, 1)

            # pad and slice the inputs if sp > 1
            if self.ulysses_sequence_parallel_size > 1:
                raise NotImplementedError("Does not support ulysses_sequence_parallel_size")


            # only pass input_ids and position_ids to enable flash_attn_varlen
            last_token_indices = [cu_seqlens[i+1] - 1 for i in range(len(cu_seqlens) - 1)]

            output = self.reward_module(input_ids=input_ids_rmpad,
                                        attention_mask=None,
                                        position_ids=position_ids_rmpad)  # prevent model thinks we are generating
            reward_rmpad = output.logits[:, last_token_indices, :]
            reward_rmpad = self._make_reward_score(reward_rmpad)

            # extract the result of the last valid token
            return reward_rmpad

    def _make_reward_score(self, logits):
        index = [14004, 8996] # token id for YES and NO
        reward = torch.softmax(logits[:, :, index], dim=-1)
        reward = torch.where(reward[:,:, 0] > 0.9, 1., 0.).squeeze(0)
        # TODO: add format reward
        return reward