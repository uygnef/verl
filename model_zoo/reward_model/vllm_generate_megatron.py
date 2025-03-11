import os
import importlib

from verl import DataProto
from verl.single_controller.base.decorator import register, Dispatch
from verl.single_controller.base.megatron.worker import MegatronWorker

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vllm_generate_fsdp.py'))
spec = importlib.util.spec_from_file_location('reward_model_base', module_path)
custom_model_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_model_base)
RewardModelWorkerBase = custom_model_base.CustomRewardModelWorker


class CustomRewardModelWorker(MegatronWorker):
    """
    reward model using text-generation hosted on VLLM
    """

    def __init__(self, config):
        # super().__init__()
        self.worker = RewardModelWorkerBase(config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.worker._init_model()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        return self.worker._compute_rm_score(data)
