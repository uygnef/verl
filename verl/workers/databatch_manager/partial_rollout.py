from recipe.partial_rollout.replay_buffer import DistributedReplayBuffer
import ray

class DataBatchManager:
    def __init__(self):
        self.replay_buffer = DistributedReplayBuffer.remote()

    def get_batch(self, batch_size):
        buffer_size = ray.get(self.replay_buffer.get_sample_nums.remote())
        print(f"get buffer_size", buffer_size)
        total_size = buffer_size - buffer_size % batch_size
        return total_size // batch_size

