import logging
import os
import warnings

import torch
import torch.distributed

from typing import List

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_init_weight_context_manager

from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))


class CustomRewardModelWorker(Worker):
    """
    reward model using text-generation hosted on VLLM
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group()
        self.config = config

        self.n_gpus_per_node = self.config.generate.get('n_gpus_per_node', 8)
        self.is_first_rank_in_node = self.rank % self.n_gpus_per_node == 0

        ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        assert ulysses_sequence_parallel_size == 1, 'RM does not support sequence parallel'

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

        self.prompt_length = self.config.generate.get('prompt_length', 512)
        self.response_length = self.config.generate.get('response_length', 512)
        ## first get from config
        ## reward model prompts may be longer than the input prompt plus response
        self.max_prompt_length = self.config.generate.get('max_prompt_length',
                                                          2 * (self.prompt_length + self.response_length))
        ## raise error on truncation
        self.truncation = self.config.generate.get('truncation', 'error')
        # the number of batches of decoded responses to print to the console
        self.num_examine = self.config.get('num_examine', 1)

        # custom function to build the prompt and compute score
        custom_fn_file = self.config.generate.custom_fn_file
        # tmp
        if custom_fn_file == 'gsm8k_rm':
            from verl.utils.reward_score.gsm8k_rm import build_rm_prompt, compute_score

            self._build_prompt_fn = build_rm_prompt
            self._compute_score_fn = compute_score
        else:
            raise NotImplementedError(f'custom_fn_file: {custom_fn_file} is not supported')

    def _build_model(self, config):
        # the following line is necessary
        from verl.utils.model import print_model_size, update_model_config, get_generation_config
        from transformers import AutoModelForCausalLM, AutoConfig

        log_gpu_memory_usage('Before init RM from HF AutoModel', logger=None)

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)

        trust_remote_code = config.model.get('trust_remote_code', False)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        input_tokenizer_path = config.model.get('input_tokenizer', None)
        if input_tokenizer_path is not None and input_tokenizer_path != local_path:
            print(f'input_tokenizer_path not equal reward model tokenizer: {input_tokenizer_path}')
            self.input_tokenizer = hf_tokenizer(input_tokenizer_path, trust_remote_code=trust_remote_code)
        else:
            self.input_tokenizer = self.tokenizer

        # override model kwargs
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {model_config}')

        init_context = get_init_weight_context_manager(use_meta_tensor=not self.is_first_rank_in_node)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reward_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                 config=model_config,
                                                                 trust_remote_code=trust_remote_code)

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(reward_module)

        log_gpu_memory_usage('After init RM from HF AutoModel', logger=None)

        return reward_module, model_config

    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.generate.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}')
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])

        rollout_name = self.config.generate.name
        if rollout_name == 'sglang':
            from verl.workers.rollout.sglang_rollout import SGLangRollout
            from verl.workers.sharding_manager.fsdp_sglang_multi_node import MultiNodeSGLangShardingManager
            log_gpu_memory_usage(f'Before building {rollout_name} reward model', logger=None)
            rm_rollout = SGLangRollout(actor_module=self.config.model.path,
                                       config=self.config.generate,
                                       tokenizer=self.tokenizer,
                                       model_hf_config=self.reward_model_config,
                                       n_gpus_per_node=self.n_gpus_per_node)
            torch.cuda.empty_cache()
            log_gpu_memory_usage(f'After building {rollout_name} reward model', logger=None)

            rm_rollout_sharding_manager = MultiNodeSGLangShardingManager(
                module=self.reward_module_fsdp,
                inference_engine=rm_rollout.inference_engine,
                model_config=self.reward_model_config,
                full_params=True,
                device_mesh=rollout_device_mesh,
                n_gpus_per_node=self.n_gpus_per_node,
                update_weights_batch_size=self.config.model.get('update_weights_batch_size', 8),
            )
            log_gpu_memory_usage('After building RM sharding manager', logger=None)
        else:
            raise NotImplementedError(f'generate name: {self.config.generate.name} is not supported')

        return rm_rollout, rm_rollout_sharding_manager

    def _init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module_fsdp, self.reward_model_config = self._build_model(config=self.config)
        self.rm_rollout, self.rm_rollout_sharding_manager = self._build_rollout()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._init_model()

    def generate(self, prompts: DataProto) -> DataProto:
        prompts = prompts.to('cuda')

        prompts.batch = prompts.batch.cuda()
        meta_info = {
            'eos_token_id':
                self.generation_config.eos_token_id
                if self.generation_config is not None else self.tokenizer.eos_token_id,
            'pad_token_id':
                self.generation_config.pad_token_id
                if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rm_rollout_sharding_manager:
            prompts = self.rm_rollout_sharding_manager.preprocess_data(prompts)
            output = self.rm_rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After RM generate', logger=None)

            output = self.rm_rollout_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After RM empty cache', logger=None)
        return output

    def prompts_to_dataproto(self, prompts: List[str]) -> DataProto:
        ## TODO: get the longest prompt length across all dp ranks, pad to the same length
        encoded = self.tokenizer(prompts,
                                 return_tensors='pt',
                                 add_special_tokens=False,
                                 padding='max_length',
                                 max_length=self.max_prompt_length,
                                 padding_side='left')
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        sequence_length = input_ids.shape[-1]
        max_length = self.max_prompt_length
        if sequence_length > max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[:, -max_length:]
                attention_mask = attention_mask[:, -max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)
        ret = DataProto.from_single_dict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        })
        return ret

    def _compute_rm_score(self, data: DataProto):
        prompts = data.non_tensor_batch['raw_prompt']
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        sequences_str = self.input_tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']

        assert len(prompts) == len(sequences_str) == len(ground_truth) == len(data_sources)

        rm_gen_prompts = [
            self._build_prompt_fn(solution, truth, question=prompt, tokenizer=self.tokenizer)
            for solution, truth, prompt in zip(sequences_str, ground_truth, prompts)
        ]

        rm_gen_batch = self.prompts_to_dataproto(rm_gen_prompts)
        rm_output_batch = self.generate(rm_gen_batch)

        rm_output_str = self.tokenizer.batch_decode(rm_output_batch.batch['responses'], skip_special_tokens=True)

        scores = [
            self._compute_score_fn(solution, truth, response)
            for solution, truth, response in zip(sequences_str, ground_truth, rm_output_str)
        ]
        print(f'scores: {scores}')

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        ## to token-level scores
        # print(f'valid_response_length: {valid_response_length}')
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
            ## TODO: print reward response to examine
        # print(f'reward_tensor summary: {reward_tensor.sum(dim=-1)}')

        ret = DataProto.from_single_dict({'rm_scores': reward_tensor})

        return ret

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        return self._compute_rm_score(data)
