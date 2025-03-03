sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-llab-data /nfs/ofs-llab-data 0a99c43dfdb44e63aeff5de3d7710460 nmgpu
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-llm-data /nfs/ofs-llm-ssd  b00b083f426245d1a6abbc2f0164124a nmgmlmodeltrain
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh luban-llm-dev /nfs/ofs_luban_dev a906ad025a834467b563d22f29763d58 nmgmlmodel
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh llm-dataset-release /nfs/ofs-llab-datasets 5ec740450e3a4556a5c50ecac6a69629 nmgmlmodel
MODEL_PATH=$1

set -x

export WANDB_API_KEY=53570e8dc045209e74372ddcacb86cb11cd932f0
export PYTHONPATH=$PYTHONPATH:/home/luban/Megatron-LM-v0.10.0/

dataset_name=$2
testset_name=$3
lr=$4
model_name=$5
kl_coef=$6

output_dir=/nfs/ofs-llm-ssd/user/kevinchenkai/models/$model_name-verl-grpo-$dataset_name-$lr
nnodes=2

cd /nfs/ofs-llm-ssd/user/liurui06/project/verl_llama/verl

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files=/nfs/ofs-llm-ssd/user/liurui06/verl_parquet_data/data/$dataset_name \
    data.val_files=/nfs/ofs-llm-ssd/user/liurui06/verl_parquet_data/data/$testset_name \
    data.train_batch_size=128 \
    data.val_batch_size=1312 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=/nfs/ofs-llm-ssd/models/Meta-Llama-3.1-8B-Instruct \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    critic.optim.lr=2e-5 \
    critic.model.path=/nfs/ofs-llm-ssd/models/Meta-Llama-3.1-8B-Instruct \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.megatron.tensor_model_parallel_size=2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_megatron_gsm8k_examples' \
    trainer.experiment_name='qwen2_5_0b5_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=15 \
    trainer.default_local_dir=/tmp-data/llama3.1_test \
    trainer.total_training_steps=3 $@
