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
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu, load_fsdp_model_to_gpu

from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask

from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy


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
            torch.distributed.init_process_group(backend='nccl')
        self.config = config

        world_size = torch.distributed.get_world_size()

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        assert ulysses_sequence_parallel_size == 1, 'RM does not support sequence parallel'

        self._is_offload_param = self.config.model.fsdp_config.get('param_offload', False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

        self.prompt_length = self.config.generate.get('prompt_length', 512)
        self.response_length = self.config.generate.get('response_length', 512)
        ## first get from config
        ## reward model prompts may be longer than the input prompt plus response
        self.max_prompt_length = self.config.generate.get(
            'max_prompt_length', 2 * (self.prompt_length + self.response_length)
        )
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
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
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

        torch_dtype = config.model.get('dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

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

        init_context = get_init_weight_context_manager(use_meta_tensor=False)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reward_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=model_config,
                attn_implementation='flash_attention_2',
                trust_remote_code=trust_remote_code,
            )
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            reward_module.to(torch_dtype)

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(reward_module)

        log_gpu_memory_usage('After init RM from HF AutoModel', logger=None)

        fsdp_config = self.config.model.fsdp_config
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=fsdp_config.get('wrap_policy', None))
        if self.config.generate.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None
        print(f'wrap_policy: {auto_wrap_policy}')

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        cpu_offload = CPUOffload(offload_params=self._is_offload_param)
        reward_module_fsdp = FSDP(
            reward_module,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=None,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False,
        )

        log_gpu_memory_usage('After RM FSDP init', logger=None)

        return reward_module_fsdp, model_config

    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.generate.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        )
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])

        if self.config.generate.name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout, vllm_mode
            from verl.workers.sharding_manager import FSDPVLLMShardingManager

            log_gpu_memory_usage('Before building vllm reward model', logger=None)
            local_path = copy_local_path_from_hdfs(self.config.model.path)
            if vllm_mode == 'customized':
                print(f'vllm mode: {vllm_mode}')
                rm_rollout = vLLMRollout(
                    actor_module=self.reward_module_fsdp,
                    config=self.config.generate,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.reward_model_config,
                )
            elif vllm_mode == 'spmd':
                print(f'vllm mode: {vllm_mode}')
                rm_rollout = vLLMRollout(
                    model_path=local_path,
                    config=self.config.generate,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.reward_model_config,
                    device_mesh=rollout_device_mesh,
                )
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")
            log_gpu_memory_usage('After building vllm reward model', logger=None)

            rm_rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.reward_module_fsdp,
                inference_engine=rm_rollout.inference_engine,
                model_config=self.reward_model_config,
                full_params='hf' in self.config.generate.load_format,
                device_mesh=rollout_device_mesh,
            )
            log_gpu_memory_usage('After building sharding manager', logger=None)
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

        log_gpu_memory_usage('Before RM generate loading model', logger=None)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module_fsdp)
        log_gpu_memory_usage('After RM generate loading model', logger=None)

        prompts.batch = prompts.batch.cuda()
        meta_info = {
            'eos_token_id': self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            'pad_token_id': self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rm_rollout_sharding_manager:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.reward_module_fsdp)
            log_gpu_memory_usage('After entering RM rollout sharding manager', logger=None)

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
        encoded = self.tokenizer(
            prompts,
            return_tensors='pt',
            add_special_tokens=False,
            padding='max_length',
            max_length=self.max_prompt_length,
            padding_side='left',
        )
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
        ret = DataProto.from_single_dict(
            {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
        )
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
        print(f'rm_gen_batch: {rm_gen_batch}')
        rm_output_batch = self.generate(rm_gen_batch)
        print(f'rm_output_batch: {rm_output_batch}')

        rm_output_str = self.tokenizer.batch_decode(rm_output_batch.batch['responses'], skip_special_tokens=True)
        print(f'rm_output_str: {rm_output_str}')

        scores = [
            self._compute_score_fn(solution, truth, response)
            for solution, truth, response in zip(sequences_str, ground_truth, rm_output_str)
        ]
        print(f'scores: {scores}')

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        ## to token-level scores
        print(f'valid_response_length: {valid_response_length}')
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
            ## TODO: print reward response to examine
        print(f'reward_tensor summary: {reward_tensor.sum(dim=-1)}')

        ret = DataProto.from_single_dict({'rm_scores': reward_tensor})

        return ret

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        return self._compute_rm_score(data)
