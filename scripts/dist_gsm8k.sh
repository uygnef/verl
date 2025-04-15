# sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-llab-data /nfs/ofs-llab-data 0a99c43dfdb44e63aeff5de3d7710460 nmgpu
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-llm-data /nfs/ofs-llm-ssd  b00b083f426245d1a6abbc2f0164124a nmgmlmodeltrain
# sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh luban-llm-dev /nfs/ofs_luban_dev a906ad025a834467b563d22f29763d58 nmgmlmodel
# sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh llm-dataset-release /nfs/ofs-llab-datasets 5ec740450e3a4556a5c50ecac6a69629 nmgmlmodel
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount_base.sh ofs-llab-volume /nfs/ofs-llab-volume b1e837d84f284c3392c4e16065fad32e nmgllab

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH=$PYTHONPATH:/home/luban/Megatron-LM-v0.11.0
output_dir=/nfs/volume-1615-1/lilinchuan/tmp-data/grpo_out
export TENSORBOARD_DIR=/nfs/volume-1615-1/lilinchuan/tmp-data/tb/qwen2_7b_rm_sglang_1
export VERL_PPO_LOGGING_LEVEL=DEBUG

#model_path=/nfs/volume-1615-2/models/Meta-Llama-3.1-8B-Instruct
model_path=/nfs/ofs-llm-ssd/models/xiamixue/ppo/r1-14b-writing-v0-stage3-2/110_0.8046875
rm_model_path=/nfs/ofs-llm-ssd/models/xiamixue/ppo/r1-14b-writing-v0-stage3-2/110_0.8046875
#RAY_ADDRESS='http://127.0.0.1:8265' ray job submit -- python3 -m recipe.gan.main_gan \
pip install torchdata --no-deps
cd /nfs/ofs-llab-volume/users/fengyu/new_verl/gan_verl
if [[ ! -z "$DISTRIBUTED_TASK_ROLE" ]] && [[ "$DISTRIBUTED_TASK_ROLE" == "master" ]];then
  DISTRIBUTED_NODE_RANK_TMP=$VC_TASK_INDEX
  echo "master, node_rank: $DISTRIBUTED_NODE_RANK_TMP"
  ray start --head --port=$PET_MASTER_PORT
  echo "ray start master in port: $PET_MASTER_PORT"

  TARGET_NODES=$DISTRIBUTED_NODE_COUNT

  while true; do
    CURRENT_NODES=$(ray list nodes | grep "ALIVE" | wc -l)

    echo "当前节点数: $CURRENT_NODES, 目标节点数: $TARGET_NODES"

    if [ "$CURRENT_NODES" -ge "$TARGET_NODES" ]; then
      echo "已达到目标节点数，开始提交任务..."
      break
    fi

    echo "等待更多节点加入..."
    sleep 10
  done
  echo "执行任务提交..."

  # Define experiment name for this stage
  ray job submit --address="http://127.0.0.1:8265" \
  -- python3 -m recipe.gan.main_gan \
    --config-path=./config --config-name='grpo_fsdp_rm_vllm' \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=/nfs/ofs-llab-volume/users/fengyu/data/train.parquet \
    data.val_files=/nfs/ofs-llab-volume/users/fengyu/data/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=7680 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.model.use_remove_padding=True \
    reward_model.model.enable_gradient_checkpointing=True \
    reward_model.actor.optim.lr=1e-6 \
    reward_model.actor.ppo_mini_batch_size=16 \
    reward_model.actor.use_dynamic_bsz=True \
    reward_model.actor.ppo_max_token_len_per_gpu=7680 \
    reward_model.actor.use_kl_loss=False \
    reward_model.actor.fsdp_config.param_offload=True \
    reward_model.actor.fsdp_config.optimizer_offload=True \
    reward_model.rollout.tensor_model_parallel_size=2 \
    reward_model.rollout.name=sglang \
    reward_model.rollout.gpu_memory_utilization=0.4 \
    reward_model.rollout.n=1 \
    +reward_model.update_interval=1 \
    reward_model.rollout.enforce_eager=True \
    reward_model.rollout.free_cache_engine=True \
    reward_model.ref.fsdp_config.param_offload=True \
    reward_model.model.path=${rm_model_path} \
    reward_model.has_validate_fn=True \
    reward_model.generate.custom_fn_file='gsm8k_rm' \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_gsm8k_rm' \
    trainer.experiment_name='qwen2_7b_rm_sglang_1' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@
    $@

  LATEST_CHECKPOINT=$(ls -d ${CHECKPOINT_PATH}/actor/global_step_* | sort -V | tail -n 1)
  MODEL_PATH=${LATEST_CHECKPOINT}
  echo "Latest checkpoint: $MODEL_PATH"
else
  DISTRIBUTED_NODE_RANK_TMP=$(($VC_TASK_INDEX+1))
  echo "worker, node_rank:$DISTRIBUTED_NODE_RANK_TMP"
  ray start --address="$VC_MASTER_HOSTS:$PET_MASTER_PORT"
  echo "ray start worker join master $VC_MASTER_HOSTS:$PET_MASTER_PORT"
  #ray start --address="$VC_MASTER_HOSTS:13001"
  #echo "ray start worker join master $VC_MASTER_HOSTS:13001"
  sleep infinity
fi