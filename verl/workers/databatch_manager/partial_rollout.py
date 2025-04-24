from recipe.partial_rollout.replay_buffer import DistributedReplayBuffer
import ray
import torch

class DataBatchManager:
    def __init__(self):
        self.replay_buffer = DistributedReplayBuffer.remote()

    def get_batch(self, batch_size):
        buffer_size = ray.get(self.replay_buffer.get_sample_nums.remote())
        print(f"get buffer_size", buffer_size)
        total_size = buffer_size - buffer_size % batch_size
        return total_size // batch_size


    def get_none_tensor(self, data, none_tensor_data):
        new_none_tensor_data = {}
        index = torch.squeeze(data['sample_id'], axis=-1).tolist()
        for i in none_tensor_data:
            new_none_tensor_data[i] = none_tensor_data[i][index]
        return new_none_tensor_data