import queue
import time
from collections import defaultdict
from typing import List

import numpy as np
from tensordict import TensorDict

from recipe.partial_rollout.replay_buffer import DistributedReplayBuffer
import ray
import torch

from verl import DataProto
from verl.utils.torch_functional import pad_2d_list_to_length, get_response_mask


class DataBatchManager:
    def __init__(self, config, tokenizer):
        '''
        Data Batch Manager handle data batch from rollout server.

        rollout string -> continue -> replay buffer
                       -> finished -> data batch manger -> tokenizer, mask
        '''


        self.replay_buffer: DistributedReplayBuffer = DistributedReplayBuffer.remote()
        self.prompt_store = {}
        self.respond_store = defaultdict(list)
        self.result = queue.Queue()
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def init_data_batch(self, batch: DataProto, rate: List[int]):
        for i, uid in enumerate(batch.non_tensor_batch['uid']):
            self.prompt_store[uid] = {"prompt": batch.non_tensor_batch['full_prompts'][i], "respond": [], "sample_id": i}
        self.data_batch = batch
        actor_data_proto, rollout_data_proto = batch.split(rate)
        return actor_data_proto, rollout_data_proto

    def preprocess(self):
        continue_data = ray.get(self.replay_buffer.get_all_data.remote(finish=False))
        partial_prompts, uid, offset = [],  [], []
        for i in continue_data:
            id = i['uid']
            i['full_prompt'] = self.prompt_store[id]['prompt'] + i['respond']
            self.prompt_store[id]['respond'].append(i['respond'])
            offset.append(len(self.prompt_store[id]['respond']) - 1)
            partial_prompts.append(i['full_prompt'])
            uid.append(id)
        return partial_prompts, uid, offset

    def postprocess(self):
        finish_data = ray.get(self.replay_buffer.get_all_data.remote(finish=True))
        if not finish_data:
            print("no data in finish queue")
            for k, v in self.respond_store.items():
                print(f"left key: {k}, len: {len(v)}")
            time.sleep(1)
            return
        for data in finish_data:
            uid = data['uid']
            if 'partial' not in data or data['partial'] == '0': # finish without cut
                self.respond_store[uid].append(data['respond'])
                # print(f"without partial: {data['respond']}")

            else:
                self.respond_store[uid].append(self.tokenizer.encode(self.prompt_store[uid]['respond'][data['offset']]) + data['respond'])
                # print(f"uid {uid}, part: {self.tokenizer.encode(self.prompt_store[uid]['respond'][data['offset']])}, part2: {data['respond']}")
            if len(self.respond_store[uid]) == self.config.actor_rollout_ref.rollout.n:
                self.result.put({'uid': uid,  'respond': self.respond_store[uid]})
                del self.respond_store[uid]


    def get_batch(self, batch_size):
        while self.result.qsize() < batch_size:
            print(f"self.result.qsize() {self.result.qsize()}")
            self.postprocess()

        responses, uids, self.uids = [], [], []
        for _ in range(batch_size):
            data = self.result.get()
            for respond in data['respond']:
                responses.append(respond)
                uids.append(self.prompt_store[data['uid']]["sample_id"])
                self.uids.append(data['uid'])
        self.uids_index = uids
        return self.make_dataproto(responses, uids, self.tokenizer.eos_token_id)



    def make_dataproto(self, response, uid_index, eos_token_id):
        response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=self.config.actor_rollout_ref.rollout.response_length)
        batch_size = len(uid_index)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device='cpu')
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        attention_mask = self.data_batch.batch['attention_mask'][uid_index]
        prompts =  self.data_batch.batch['input_ids'][uid_index]
        position_ids = self.data_batch.batch['position_ids'][uid_index]
        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask.cpu(), response_attention_mask.cpu()), dim=-1)
        seq = torch.cat([prompts.cpu(), torch.tensor(response)], dim=-1)
        # all the tp ranks should contain the same data here. data in all ranks are valid

        batch = TensorDict({
                'prompts': prompts.cpu(),
                'responses': response.cpu(),
                'input_ids': seq.cpu(),  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask.cpu(),
                'position_ids': position_ids.cpu(),
        }, batch_size=batch_size)
        data = DataProto()
        data.batch = batch
        return data


    def get_none_tensor(self, none_tensor_data):
        new_none_tensor_data = {}
        for i in none_tensor_data:
            new_none_tensor_data[i] = none_tensor_data[i][self.uids_index]
        new_none_tensor_data["uid"] = np.array(self.uids)
        return new_none_tensor_data
