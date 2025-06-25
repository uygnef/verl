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
"""
An naive implementation of split placment example
"""
import time
import uuid
from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_throughout_metrics
from verl.trainer.ppo.ray_trainer import compute_advantage, apply_kl_penalty, reduce_metrics, compute_data_metrics, \
    _timer, compute_timing_metrics


def fit(self, batch_size):
    """
    The training loop of PPO.
    The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
    The light-weight advantage computation is done on the driver process.
    """
    from verl.utils.tracking import Tracking
    from omegaconf import OmegaConf

    logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

    self.global_steps = 0

    # load checkpoint before doing anything
    self._load_checkpoint()

    # perform validation before training
    # currently, we only support validation using the reward_function.
    if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        val_metrics = self._validate()
        assert val_metrics, f"{val_metrics=}"
        pprint(f"Initial validation metrics: {val_metrics}")
        logger.log(data=val_metrics, step=self.global_steps)
        if self.config.trainer.get("val_only", False):
            return
    # add tqdm
    progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

    # we start from step 1
    self.global_steps += 1
    last_val_metrics = None
    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            metrics = {}
            timing_raw = {}

            batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                     dtype=object)
            # pop those keys for generation
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "uid" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("uid")
            if "full_prompts" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("full_prompts")

            gen_batch = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            gen_batch.batch['sample_id'] = torch.unsqueeze(
                torch.range(0, list(batch.batch.batch_size)[0] - 1, dtype=torch.long), dim=-1)
            # gen_batch.non_tensor_batch["uid"] = batch.non_tensor_batch["uid"]
            is_last_step = self.global_steps >= self.total_training_steps

            with _timer('step', timing_raw):
                gen_batch1, gen_batch2 = self.databatch_manager.init_data_batch(gen_batch, [1,1])

                # generate a batch
                with _timer('gen', timing_raw):
                    print(f"start batch: gen_batch size {gen_batch.batch.batch_size}, "
                          f"raw_prompt_ids batch size {len(gen_batch.non_tensor_batch['raw_prompt_ids'])}")
                    # a = self.actor_rollout_wg._broadcast_to_vllm()
                    # self.rollout_wg.rollout_broadcast_to_vllm()
                    # a = ray.get(a)
                    prompts = gen_batch1.non_tensor_batch['raw_prompt']
                    uid = gen_batch1.non_tensor_batch['uid']
                    self.async_rollout_manager.generate_sequences(prompts, uid)

                    print(f"gen_batch2 {gen_batch2}")
                    result = self.actor_rollout_wg.generate_sequences(gen_batch2)
                    print(f"actor output {result}")
                    # partial rollout
                    partial_prompts, partial_uid, offset = self.databatch_manager.preprocess()
                    print(f"partial_prompts: {partial_prompts}, partial_uid: {partial_uid}, offset: {offset}")
                    self.async_rollout_manager.generate_sequences(partial_prompts, partial_uid, offset, 1, self.config.actor_rollout_ref.rollout.response_length - 200)

                total_batch_nums = 0
                while total_batch_nums < batch_size:
                    # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # if self.config.trainer.balance_batch:
                    #     self._balance_batch(batch, metrics=metrics)

                    # batch_nums = self.databatch_manager.get_batch_num(
                    #     batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
                    # print(f"get_batch_num nums {batch_nums} batch size", total_batch_nums)
                    # if batch_nums == 0:
                    #     time.sleep(3)
                    #     print("0 batch num sleep 3 seconds")
                    #     continue
                    # total_batch_nums += batch_nums

                    with _timer('get_batch', timing_raw):
                        print(f"get batch data {total_batch_nums} vs {batch_size}")
                        data = self.databatch_manager.get_batch(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
                        total_batch_nums += self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                        print(f"after get batch data {total_batch_nums} vs {batch_size}")
                        print(f"data batch keys : {data.batch.keys()}")
                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(data)
                        new_none_tensor_batch = self.databatch_manager.get_none_tensor(batch.non_tensor_batch)
                        micro_batch = data.union(old_log_prob)
                        micro_batch.non_tensor_batch = new_none_tensor_batch
                        print(f"old_log_prob data batch keys : {micro_batch.batch.keys()}")

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(micro_batch)
                            micro_batch = micro_batch.union(ref_log_prob)
                        print(f"ref_log_prob data batch keys : {micro_batch.batch.keys()}")

                    # compute values
                    # if self.use_critic:
                    #     with _timer('values', timing_raw):
                    #         values = self.critic_wg.compute_values(batch)
                    #         batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        # if self.use_rm:
                        #     # we first compute reward model score
                        #     reward_tensor = self.rm_wg.compute_rm_score(batch)
                        #     batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(micro_batch)
                        micro_batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            micro_batch, kl_metrics = apply_kl_penalty(micro_batch, kl_ctrl=self.kl_ctrl_in_reward,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            micro_batch.batch["token_level_rewards"] = micro_batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        micro_batch = compute_advantage(micro_batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                        print(f"compute_advantage data batch keys : {micro_batch.batch.keys()}")


                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(micro_batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                        print(f"critic_output data batch keys : {micro_batch.batch.keys()}")


                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(micro_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)


                with _timer("sync_rollout_weights", timing_raw):
                    self.actor_rollout_wg._broadcast_to_vllm()
                    self.rollout_wg.rollout_broadcast_to_vllm()

                # validate
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with _timer('testing', timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (is_last_step or
                                                          self.global_steps % self.config.trainer.save_freq == 0):
                    with _timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()
                # collect metrics
                micro_batch.meta_info['global_token_num'] = torch.sum(micro_batch.batch['attention_mask'], dim=-1).tolist()
                metrics.update(compute_data_metrics(batch=micro_batch, use_critic=self.use_critic))
                print(f"compute_timing_metrics data batch keys : {micro_batch.batch.keys()}")
                metrics.update(compute_timing_metrics(batch=micro_batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=micro_batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f'Final validation metrics: {last_val_metrics}')
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1