import os
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from tensordict import TensorDict
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.test_utilities import Utils
import torch.nn as nn
from model_zoo.reward_model.openmath import OpenmathRewardModel
from verl import DataProto


class TestMegatronModel:

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )

    def test_reward_model(self):

        # 模拟tokenizer
        class MockTokenizer:
            def __init__(self):
                self.pad_token_id = 0

            def decode(self, input_ids):
                return "mock decoded string"

            def __call__(self, text, return_tensors='pt'):
                return {'input_ids': torch.tensor([[1, 2, 3]])}

        sft_tokenizer = MockTokenizer()
        rm_tokenizer = MockTokenizer()

        # 模拟配置
        class Config:
            def __init__(self):
                self.param_offload = False
                self.ppo_micro_batch_size_per_gpu = 1

            def __contains__(self, item):
                return item in self.__dict__

        config = Config()

        # 模拟模型配置
        class ModelConfig:
            def __init__(self):
                self.hidden_size = 768

        model_config = ModelConfig()
        # 模拟Megatron配置
        class MegatronConfig:
            def __init__(self):
                self.sequence_parallel = False
            def __contains__(self, item):
                return item in self.__dict__
        megatron_config = MegatronConfig()
        # 创建OpenmathRewardModel实例
        model = OpenmathRewardModel(
            config=config,
            model_config=model_config,
            reward_model_module=nn.ModuleList([self.gpt_model]),
            megatron_config=megatron_config,
            sft_tokenizer=sft_tokenizer,
            rm_tokenizer=rm_tokenizer
        )

        # 模拟输入数据
        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long).cuda()
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).cuda()
        responses = torch.randint(0, 1000, (batch_size, seq_len)).cuda()

        batch = TensorDict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'responses': responses
        }, batch_size=batch_size)

        data = DataProto(batch=batch)

        # 调用compute_reward方法
        result = model.compute_reward(data)

        # 打印输出
        print("Compute reward result:", result.batch)

        # 清理分布式环境
        mpu.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    a = TestMegatronModel()
    a.setup_method()
    a.test_reward_model()