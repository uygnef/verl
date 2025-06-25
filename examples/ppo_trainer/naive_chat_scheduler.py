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
import asyncio
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from recipe.partial_rollout.replay_buffer import DistributedReplayBuffer
from verl.protocol import DataProto
from verl.workers.rollout.async_server import ChatCompletionScheduler
import ray

class NaiveChatCompletionScheduler(ChatCompletionScheduler):
    """
    A very naive implementation of ChatCompletionScheduler for demo purpose,
    only do single-turn chat completion.
    """
    def set_replay_buffer(self, replay_buffer: DistributedReplayBuffer):
        self.replay_buffer = replay_buffer

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[NaiveChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            assert exception is None, f"exception: {exception}"
            conversation, batch_conversations, batch_index = (
                info["conversation"],
                info["batch_conversations"],
                info["batch_index"],
            )

            conversations = []
            for choice in completions.choices:
                chat = conversation.copy()
                chat.append({"role": choice.message.role, "content": choice.message.content})
                conversations.append(chat)
            batch_conversations[batch_index] = conversations

            # NOTE: we can call tools and resubmit chat completions here.
            # call_tools(completions, info)
            # await self.submit_chat_completions(callback2, ...)

        # TODO: we may need to control max concurrent requests here, or it will harm prefix cache hit rate.
        tasks, batch_conversations = [], [None] * len(batch)
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_index": batch_index,
                            "conversation": list(conversation),
                        },
                        model=self.model_name,
                        messages=conversation.tolist(),
                        **kwargs,
                    )
                )
            )
        await asyncio.gather(*tasks)
        print("[NaiveChatCompletionScheduler] generate_sequences done")

        return self._postprocess(batch, batch_conversations, kwargs["n"])

    async def generate_sequences_async(
            self, prompts, uid, offset=None, rollout_n=None, response_length=None, **sampling_params
    ):
        kwargs = dict(
            n= rollout_n if rollout_n else self.config.n,
            max_completion_tokens=self.config.response_length if response_length is None else response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        if offset:
            print(f"offset is : {offset}, prompts : {prompts}, response length: {response_length}, rollout_n: {rollout_n}, uid: {uid}")
        kwargs.update(sampling_params)
        print(f"[NaiveChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            assert exception is None, f"exception: {exception}"

            print(f"id {info['batch_index']},choice.message.content: {completions.choices}, len: {len(completions.choices)}, "
                  f"result {[choice.message.content for choice in completions.choices]}", flush=True)
            for i, choice in enumerate(completions.choices):
                print(f"offset: {offset}, info['batch_index']: {info['batch_index']}, choice.message.content: {choice.message.content}")
                data = {
                    "uid": uid[info["batch_index"]],
                    "respond": self.tokenizer.encode(choice.message.content),
                    "offset": offset[info["batch_index"]] if offset is not None else i,
                    "partial": '1' if offset is not None else '0',
                }
                print(f"_postprocess_item: {data}, ", flush=True)
                ray.get(self.replay_buffer.put_item.remote(data, finished=True))

            # return info["batch_index"], [choice.message.content for choice in completions.choices]

        tasks = []
        batch_conversations = [None] * len(prompts)
        for batch_index, conversation in enumerate(prompts):
            extra_headers = {}
            if offset is not None:
                print(f"before: partial batch_index {batch_index},  conversation {conversation}")
                conversation = np.array(self.convert_text_to_messages(conversation))
                print(f"after: partial batch_index {batch_index}, conversation {conversation}, type {type(conversation[0])}")
            else:
                print(f"origin batch_index {batch_index},  conversation {conversation}, type {type(conversation[0])}")


            extra_headers['x-request-id'] = uid[batch_index]
            task = asyncio.create_task(
                self.submit_chat_completions(
                    callback=callback,
                    callback_additional_info={
                        "batch_index": batch_index,
                    },
                    model=self.model_name,
                    messages=conversation.tolist(),
                    extra_headers=extra_headers,
                    **kwargs,
                )
            )
            tasks.append(task)

        for coro in asyncio.as_completed(tasks):
            await coro
            if offset is not None:
                print("partial finish")
        print("[NaiveChatCompletionScheduler] generate_sequences done")

    def convert_text_to_messages(self, text):
        messages = []
        lines = text.split('\n')  # 按行分割

        current_role = None
        current_content = []

        for line in lines:
            if line.startswith('<|im_start|>'):
                # 遇到新角色，保存之前的内容
                if current_role is not None:
                    messages.append({
                        "role": current_role,
                        "content": '\n'.join(current_content).strip()
                    })
                    current_content = []

                # 解析新角色
                current_role = line.split('|')[2][1:]  # 提取 system/user/assistant
            elif line.startswith('<|im_end|>'):
                continue  # 忽略结束标记
            else:
                current_content.append(line)  # 收集内容

        # 添加最后一个消息块
        if current_role and current_content:
            messages.append({
                "role": current_role,
                "content": '\n'.join(current_content).strip()
            })

        return messages

    def _postprocess_item(self, responds, uid, offset):
        # batch id: idx of the batch
        # respond: respond
        # offset: for grpo, idx inner group
        for i, respond in enumerate(responds):
            data = {
                "uid": uid,
                "respond": respond,
                "offset": offset if offset else i
            }
            print(f"_postprocess_item: {data}", flush=True)
            ray.get(self.replay_buffer.put_item.remote(data, finished=True))


    def _postprocess(self, batch: DataProto, batch_conversations: List[List[List[Dict[str, str]]]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in batch.non_tensor_batch["raw_prompt"]]

        # flatten batch_conversations if n > 1
        assert len(batch_conversations) == len(prompts)
        batch_conversations = [conversation for conversations in batch_conversations for conversation in conversations]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False) for conversation in batch_conversations]

        # responses: [response]
        # TODO: mask out tools calling tokens?
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],
                "responses": responses["input_ids"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(input_ids),
        )

        return DataProto(batch=batch)

