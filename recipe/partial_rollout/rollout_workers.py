# Copyright 2025 xx and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings
import psutil

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoConfig

import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_model_to_cpu, load_fsdp_optimizer, \
    load_fsdp_model_to_gpu
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from codetiming import Timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

from verl.workers.fsdp_workers import create_device_mesh


class PartialRolloutWorker(Worker):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group()

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)
        # TODO(fengyu): does support ulysses_sequence_parallel_size?
        self.ulysses_sequence_parallel_size = 1
        from verl.utils.model import get_generation_config
        self.generation_config = get_generation_config(self.config.model.path, True)

        # normalize rollout config
        if self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (self.device_mesh.size() //
                                                               self.ulysses_sequence_parallel_size)
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size

    def _build_rollout(self, replay_buffer=None):
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=('dp', 'infer_tp'))
        rollout_name = self.config.rollout.name

        if rollout_name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout, vllm_mode
            from verl.workers.sharding_manager import FSDPVLLMShardingManager
            log_gpu_memory_usage(f'Before building {rollout_name} rollout', logger=None)
            local_path = copy_to_local(self.config.model.path)
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
            actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)

            if vllm_mode == 'spmd':
                rollout = vLLMRollout(model_path=local_path,
                                      config=self.config.rollout,
                                      tokenizer=self.tokenizer,
                                      model_hf_config=actor_model_config,
                                      device_mesh=rollout_device_mesh,
                                      replay_buffer=replay_buffer)
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")
            log_gpu_memory_usage(f'After building {rollout_name} rollout', logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            from verl.workers.sharding_manager.fsdp_sglang import PartialRolloutManager
            rollout_sharding_manager = PartialRolloutManager(inference_engine=rollout.inference_engine,
                                                               full_params='hf' in self.config.rollout.load_format,
                                                               device_mesh=rollout_device_mesh)
            log_gpu_memory_usage('After building sharding manager', logger=None)

        elif rollout_name == 'sglang':
            from verl.workers.rollout.sglang_rollout import SGLangRollout
            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to SGLang's model_runner would check CUDA device capability.
            # However, due to veRL's setting, the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.fsdp_sglang import PartialRolloutManager
            log_gpu_memory_usage(f'Before building {rollout_name} rollout', logger=None)
            rollout = SGLangRollout(actor_module=self.config.model.path,
                                    config=self.config.rollout,
                                    tokenizer=self.tokenizer,
                                    model_hf_config=self.actor_model_config)
            log_gpu_memory_usage(f'After building {rollout_name} rollout', logger=None)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            rollout_sharding_manager = PartialRolloutManager(module=self.actor_module_fsdp,
                                                                 inference_engine=rollout.inference_engine,
                                                                 model_config=self.actor_model_config,
                                                                 full_params='hf' in self.config.rollout.load_format,
                                                                 device_mesh=rollout_device_mesh)
            log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.rollout, self.rollout_sharding_manager = self._build_rollout()


    @register(dispatch_mode=Dispatch.DP_DISPATCH_ONLY)
    def generate_sequences(self, prompts: DataProto, blocking=False):
        # Support all hardwares
        prompts = prompts.to(torch.cuda.current_device())

        meta_info = {
            'eos_token_id':
                self.generation_config.eos_token_id
                if self.generation_config is not None else self.tokenizer.eos_token_id,
            'pad_token_id':
                self.generation_config.pad_token_id
                if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:

            # after parameters sync with rollout, offload actor model to CPU
            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            log_gpu_memory_usage('After rollout generation', logger=logger)

            # output = self.rollout_sharding_manager.postprocess_data(output)

        # output = output.to('cpu')

        # clear kv cache
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output
