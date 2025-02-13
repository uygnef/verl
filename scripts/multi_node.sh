MODEL_PATH=xxx/models/Qwen2.5-7B
export VLLM_ATTENTION_BACKEND=XFORMERS
output_dir=XXX
source /opt/miniconda3/bin/activate verl

cd xxx/verl
if [[ ! -z "$DISTRIBUTED_TASK_ROLE" ]] && [[ "$DISTRIBUTED_TASK_ROLE" == "master" ]];then
  DISTRIBUTED_NODE_RANK_TMP=$VC_TASK_INDEX
  echo "master, node_rank: $DISTRIBUTED_NODE_RANK_TMP"
  ray start --head --port=$PET_MASTER_PORT
  echo "ray start master in port: $PET_MASTER_PORT"
  ray job submit --address="http://127.0.0.1:8265" \
   -- python -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files=XXX \
      data.val_files=XXX \
      data.dataset=LogicRLDataset \
      data.train_batch_size=8 \
      data.val_batch_size=8 \
      data.max_prompt_length=400 \
      data.max_response_length=2048 \
      actor_rollout_ref.model.path=$MODEL_PATH\
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=512 \
      actor_rollout_ref.actor.ppo_micro_batch_size=64 \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.kl_loss_coef=0.001 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=True \
      actor_rollout_ref.actor.fsdp_config.grad_offload=True \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
      actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
      actor_rollout_ref.rollout.n=16 \
      actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      algorithm.kl_ctrl.kl_coef=0.001 \
      trainer.critic_warmup=0 \
      trainer.logger=['console'] \
      trainer.project_name='GRPO_logic_KK_logic_lr' \
      trainer.experiment_name='Qwen-7B' \
      trainer.n_gpus_per_node=8 \
      trainer.nnodes=1 \
      trainer.default_local_dir=$output_dir \
      trainer.default_hdfs_dir=null \
      trainer.save_freq=50 \
      trainer.test_freq=50 \
      trainer.total_epochs=5
else
  DISTRIBUTED_NODE_RANK_TMP=$(($VC_TASK_INDEX+1))
  echo "worker, node_rank:$DISTRIBUTED_NODE_RANK_TMP"
  ray start --address="$VC_MASTER_HOSTS:$PET_MASTER_PORT"
  echo "ray start worker join master $VC_MASTER_HOSTS:$PET_MASTER_PORT"
  sleep infinity
fi