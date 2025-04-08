from typing import Any, Optional, Dict

import ray
import time
import unittest

import torch
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

    # def put_batch(self, batch):


    def put(self, data: Any, is_finish: bool) -> None:
        """
        将数据放入 Replay Buffer

        Args:
            data: 要存储的数据
            is_finish: 数据是否完成（包含 eos）
        """
        if is_finish:
            # 如果是完成的数据，放入 finish 队列
            self.finish_queue.put(data)
            print(f"rank : put to finish {self.finish_queue.qsize()}", flush=True)
        else:
            # 如果是未完成的数据，放入 continue 队列
            self.continue_queue.put(data)
            print(f"rank : put to continue_queue {self.continue_queue.qsize()}", flush=True)


    def get(self, consumer_type: str, batch_size: int=1) -> Optional[Any]:
        """
        从 Replay Buffer 中获取数据

        Args:
            consumer_type: 消费者类型，可以是 "actor_update" 或其他类型
            batch_size: 获取的batch size
        Returns:
            数据，如果没有数据则返回 None
        """
        data = None
        while data is None:
            try:
                if consumer_type == "actor_update":
                    data = self.finish_queue.get(timeout=20)
                else:
                    data = self.continue_queue.get(timeout=20)
            except Exception as e:
                print("EXCEPTION", flush=True)
                continue
        return data

    def empty(self) -> bool:
        return self.continue_queue.empty()


