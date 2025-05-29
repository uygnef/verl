from typing import Any, Optional, Dict

import ray
import time
import unittest

import torch
from tensordict import TensorDict


@ray.remote
class ReplayBufferLock:
    def __init__(self):
        self.locked = False

    def acquire(self) -> None:
        """获取锁"""
        while self.locked:
            pass
        self.locked = True

    def release(self) -> None:
        """释放锁"""
        self.locked = False


@ray.remote
class DistributedReplayBuffer:
    def __init__(self, max_size: int = 10000, eos_token: str = '[EOS]'):
        """
        初始化分布式 Replay Buffer

        Args:
            max_size: 回放缓冲区最大大小
            eos_token: 结束标记符
        """
        self.max_size = max_size
        self.eos_token = eos_token
        from ray.util.queue import Queue
        self.finish_queue = Queue()
        self.continue_queue = Queue()
        self.total_samples = 0

    def put_batch(self, data_list: TensorDict, finished: bool) -> None:
        """
        将数据放入 Replay Buffer

        Args:
            data: 要存储的数据
            finished: 数据是否完成（包含 eos）
        """
        if finished:
            # 如果是完成的数据，放入 finish 队列
            self.finish_queue.put_nowait(data_list)
            # print(f"rank : put to finish {self.finish_queue.qsize()}", flush=True)
        else:
            # 如果是未完成的数据，放入 continue 队列
            self.continue_queue.put_nowait(data_list)
            # print(f"rank : put to continue_queue {self.continue_queue.qsize()}", flush=True)

    def put(self, data: Any, finished: bool) -> None:
        """
        将数据放入 Replay Buffer

        Args:
            data: 要存储的数据
            finished: 数据是否完成（包含 eos）
        """
        if finished:
            # 如果是完成的数据，放入 finish 队列
            self.finish_queue.put_nowait(data)
            self.total_samples += list(data.batch_size)[0]
            print(f"rank : put {data.batch_size} to finish {self.finish_queue.qsize()}", flush=True)
        else:
            # 如果是未完成的数据，放入 continue 队列
            self.continue_queue.put_nowait(data)
            print(f"rank : put {data.batch_size} to continue_queue {self.continue_queue.qsize()}", flush=True)

    def get(self, consumer_type: str, batch_size: int = 1) -> Optional[Any]:
        """
        从 Replay Buffer 中获取数据

        Args:
            consumer_type: 消费者类型，可以是 "actor_update" 或其他类型
            batch_size: 获取的batch size
        Returns:
            数据，如果没有数据则返回 None
        """
        current_size = 0
        result_data = None
        queue = self.finish_queue if consumer_type == "actor_update" else self.continue_queue

        while batch_size > current_size:
            try:
                # Get data from the appropriate queue with timeout
                data = queue.get(timeout=20)

                # Get the actual batch size of the retrieved data
                data_size = len(data) if hasattr(data, '__len__') else 1

                if result_data is None:
                    result_data = data
                    current_size = data_size
                else:
                    # Concatenate the new data with existing result
                    result_data = torch.cat([result_data, data], dim=0)
                    current_size += data_size
            except queue.Empty:
                # Timeout occurred, return what we have (might be None or partial batch)
                break

        # If we collected more than needed, split and return excess to queue
        if current_size > batch_size and result_data is not None:
            excess = result_data[batch_size:]
            result_data = result_data[:batch_size]

            # Return excess data to the original queue
            queue.put_nowait(excess)
        if consumer_type == "actor_update":
            self.total_samples -= batch_size
        return result_data

    def empty(self) -> bool:
        return self.continue_queue.empty()


    def get_continue_size(self) -> int:
        return self.continue_queue.qsize()

    def finish_size(self):
        return self.finish_queue.qsize()

    def get_sample_nums(self):
        return self.total_samples