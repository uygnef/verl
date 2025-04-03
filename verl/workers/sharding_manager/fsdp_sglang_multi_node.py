# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
import logging
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.utils.debug import log_gpu_memory_usage
from sglang.srt.entrypoints.verl_engine import VerlEngine
from .base import BaseShardingManager
# from verl.third_party.sglang import parallel_state as sglang_ps

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class MultiNodeSGLangShardingManager(BaseShardingManager):

    def __init__(
        self,
        module,
        inference_engine: VerlEngine,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        n_gpus_per_node: int = 8,
        update_weights_batch_size: int = 8,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.update_weights_batch_size = update_weights_batch_size

        dp_size = self.device_mesh['dp'].mesh.size()[0]
        tp_size = self.device_mesh['infer_tp'].mesh.size()[0]
        world_size = dp_size * tp_size
        nnodes = world_size // n_gpus_per_node
        self.global_rank = self.device_mesh.get_rank()

        self.update_weight_groups = []
        for i in range(nnodes):
            _ranks = list(range(i * n_gpus_per_node, (i + 1) * n_gpus_per_node))
            _group = dist.new_group(backend='nccl', ranks=_ranks)
            self.update_weight_groups.append(_group)
            if self.global_rank in _ranks:
                self.update_weight_group = _group
                self.src_rank_in_node = _ranks[0]
                print(
                    f"MultiNodeSGLangShardingManager {self.global_rank=}, create local node group {i} with ranks {_ranks}, {self.src_rank_in_node=}",
                    flush=True)

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        local_rank = torch.distributed.get_rank()
        log_gpu_memory_usage('Before update weights in sharding manager memory', logger=None, rank=local_rank)

        self.module.eval()
        self.inference_engine.resume_memory_occupation()
        self.sglang_update_weights_sharded()

        dist.barrier()
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=None, rank=local_rank)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        local_rank = torch.distributed.get_rank()
        log_gpu_memory_usage('Before SGLang offload in sharding manager', logger=None, rank=local_rank)
        self.inference_engine.release_memory_occupation()
        log_gpu_memory_usage('After SGLang offload in sharding manager', logger=None, rank=local_rank)

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                            size=self.device_mesh["infer_tp"].mesh.size()[0],
                                            group=self.device_mesh["infer_tp"].get_group(),
                                            dim=0)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        global_rank = self.device_mesh.get_rank()
        tp_rank = self.device_mesh["infer_tp"].get_local_rank()
        tp_size = self.device_mesh["infer_tp"].mesh.size()[0]
        src_rank = global_rank // tp_size * tp_size
        broadcast_dict_tensor(data.batch, src=src_rank, group=self.device_mesh["infer_tp"].get_group())
        if tp_size > 1:
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[tp_rank]
        return data

    def sglang_update_weights_sharded(self):
        """
        update weights for sglang verl engine, in a sharded way
        local group is the group of ranks in the same node
        """
        param_list = list(self.module.named_parameters())
        update_batch_size = self.update_weights_batch_size
        rank = self.global_rank
        for i in range(0, len(param_list), update_batch_size):
            idx_end = min(i + update_batch_size, len(param_list))
            batch = param_list[i:idx_end]

            # names = [name for name, _ in batch]
            # log_gpu_memory_usage(f'Before update_weights for {names}', None, rank=rank)
            batch2 = []
            for name, param in batch:
                ## broadcast the model parameters
                if rank == self.src_rank_in_node:
                    param = param.detach().cuda()
                else:
                    param = torch.zeros(param.shape, dtype=param.dtype, device=torch.cuda.current_device())
                dist.broadcast(param, src=self.src_rank_in_node, group=self.update_weight_group)
                # print(f"rank {rank}, {name} {param.shape=} {param.dtype=} {param.device=}", flush=True)
                batch2.append((name, param))
            # log_gpu_memory_usage(f'After broadcast weights for {names}', None, rank=rank)

            self.inference_engine.update_weights_from_tensor(batch2, load_format=None)

            # print(f"rank {rank} update weight batch {i} before barrier", flush=True)
            dist.barrier(group=self.update_weight_group)
            # log_gpu_memory_usage(f'After update_weights for {names}', None, rank=rank)

            if rank == self.src_rank_in_node:
                for name, param in batch2:
                    param.cpu()
            else:
                del batch2
            torch.cuda.empty_cache()
            # log_gpu_memory_usage(f'After empty cache for {names}', None, rank=rank)
