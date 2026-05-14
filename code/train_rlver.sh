
set -e
set -u

# 说明：
# - 该脚本由原 7B 配置改写为 1.5B（/data/yywang/Models/Qwen2.5-1.5B-Instruct）
# - 你的硬件是单机 8×3090（24GB），因此需要把“上下文长度/每轮生成长度/每 GPU token 上限”等显存相关参数调低。
# - 不再在脚本中硬编码 W&B Token：请在运行前导出环境变量 WANDB_API_KEY 或 WANDB_TOKEN。

RUN_NAME="rlver-reproduction-qwen2.5-1.5b-grpo-thinking"
IF_THINK=True
DIR_TO_SAVE_CKPTS="/data/yywang/Projects/EvoMem/GitHub/digitalhuman/RLVER/reproduction/qwen2.5-1.5b-grpo-thinking/checkpoints"
# 屏蔽被占用的 4、7 号 GPU，仅使用 0,1,2,3,5,6
export CUDA_VISIBLE_DEVICES="0,1,2,3"
N_GPUS_PER_NODE=4
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export RAY_record_ref_creation_sites=1
export HYDRA_FULL_ERROR=1
# pip install torchdata
mkdir -p $DIR_TO_SAVE_CKPTS

# 将运行时环境（Ray Job runtime env）构造成合法 JSON，并把 WANDB key 透传给 job（如果存在）。
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

env_vars = {
    "HYDRA_FULL_ERROR": "1",
}

cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
if cuda_visible_devices:
  env_vars["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

# 兼容两种命名：WANDB_API_KEY 优先，其次 WANDB_TOKEN
wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_TOKEN")
if wandb_key:
    env_vars["WANDB_API_KEY"] = wandb_key

runtime_env = {
    "env_vars": env_vars,
    "working_dir": "./",
    "pip": ["latex2sympy2", "word2number", "timeout_decorator"],
}
print(json.dumps(runtime_env))
PY
)

# 如果没有设置 WANDB key，则默认仅输出 console 日志，避免 wandb 初始化失败。
if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_TOKEN:-}" ]]; then
  TRAINER_LOGGER="['console']"
else
  TRAINER_LOGGER="['console','wandb']"
fi

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="$RUNTIME_ENV_JSON" -- CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    +data.virtual_dataset_size=32000 \
    +data.val_virtual_dataset_size=320 \
    data.prompt_key=prompt \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=3072 \
    data.max_response_length=2048 \
    data.return_raw_chat=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.thinking=$IF_THINK \
    actor_rollout_ref.model.path="/data/yywang/Models/Qwen2.5-1.5B-Instruct" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.use_loss_generation_mask=True \
    actor_rollout_ref.rollout.name=vllm_multi_turn_via_chat \
    +actor_rollout_ref.rollout.environment.name=url_environment \
    +actor_rollout_ref.rollout.environment.per_turn_length=1024 \
    +actor_rollout_ref.rollout.environment.max_turns=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    trainer.project_name=verl \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=$DIR_TO_SAVE_CKPTS \
    trainer.logger=$TRAINER_LOGGER \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.save_rollout=True \
    trainer.test_freq=999999 \
    trainer.total_epochs=999999 \
    trainer.total_training_steps=1000 \
    2>&1 | tee -a "$DIR_TO_SAVE_CKPTS/train.log"
    # critic.optim.lr=5e-6 \
    # critic.optim.lr_warmup_steps_ratio=0.05 \
    # critic.model.path="YOUR_CRITIC_MODEL_PATH" \
    # critic.ppo_mini_batch_size=32 \
    # critic.model.use_remove_padding=True \
    # critic.model.fsdp_config.param_offload=False \
    # critic.model.fsdp_config.optimizer_offload=False \
    # critic.ppo_max_token_len_per_gpu=48000 \
    # critic.forward_max_token_len_per_gpu=48000 \