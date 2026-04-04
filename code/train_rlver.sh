
set -e
set -u

# 说明：
# - 该脚本由原 7B 配置改写为 1.5B（/data/yywang/Models/Qwen2.5-1.5B-Instruct）
# - 你的硬件是单机 8×3090（24GB），因此需要把“上下文长度/每轮生成长度/每 GPU token 上限”等显存相关参数调低。
# - 不再在脚本中硬编码 W&B Token：请在运行前导出环境变量 WANDB_API_KEY 或 WANDB_TOKEN。

RUN_NAME="rlver-reproduction-qwen2.5-1.5b-ppo-thinking"
IF_THINK=True
DIR_TO_SAVE_CKPTS="/data/yywang/Projects/EvoMem/GitHub/digitalhuman/RLVER/reproduction/qwen2.5-1.5b-ppo-thinking/checkpoints"
export RAY_record_ref_creation_sites=1
export HYDRA_FULL_ERROR=1
pip install torchdata
mkdir -p $DIR_TO_SAVE_CKPTS

# 将运行时环境（Ray Job runtime env）构造成合法 JSON，并把 WANDB key 透传给 job（如果存在）。
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

env_vars = {
    "TOKENIZERS_PARALLELISM": "true",
    "HYDRA_FULL_ERROR": "1",
}

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
    --runtime-env-json="$RUNTIME_ENV_JSON" -- PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    data.virtual_dataset_size=32000 \
    data.val_virtual_dataset_size=320 \
    data.prompt_key=prompt \
    data.train_batch_size=32 \
    data.val_batch_size=32 \  # 3090(24GB) 建议：先把总长度压到 ~5k 以内，稳定后再逐步加。 可调范围：max_prompt_length 1024~4096；max_response_length 512~3072
    data.max_prompt_length=3072 \
    data.max_response_length=2048 \
    data.return_raw_chat=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.thinking=$IF_THINK \
    actor_rollout_ref.model.path="/data/yywang/Models/Qwen2.5-1.5B-Instruct" \
    actor_rollout_ref.model.use_remove_padding=True \
    # 1.5B 一般可以比 7B 稍微更大胆一点的 lr，但没有唯一最优。
    # 可调范围（经验）：1e-6 ~ 5e-6；若 loss/奖励震荡明显，往下调。
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    # 关键显存参数：每 GPU 的 token 上限（影响 actor forward/backward 的批内 token 聚合）。
    # 参考默认配置：n * max_prompt_length + max_response_length。
    # 对于 1.5B + 3090，建议先用 16384 或更低，稳定后再提升。
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.use_loss_generation_mask=True \
    actor_rollout_ref.rollout.name=vllm_multi_turn_via_chat \
    +actor_rollout_ref.rollout.environment.name=url_environment \
    # multi-turn 每轮生成上限（vLLM SamplingParams.max_tokens）。
    # 3090 建议：512~1536；过大容易导致 vLLM KV cache 爆显存。
    +actor_rollout_ref.rollout.environment.per_turn_length=1024 \
    +actor_rollout_ref.rollout.environment.max_turns=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    # GRPO 通常需要 n>1（同一 prompt 采样多条回复）。3090 上建议先从 2 起。
    # 可调范围：2~4（越大越吃显存/吞吐越慢）。
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    # vLLM 的显存占用比例（越高越容易 OOM，尤其是 24GB 卡）。
    # 可调范围：0.45~0.7
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    # 当前配置禁用 KL（actor.use_kl_loss=False 且 kl_coef=0），RefPolicy 会被跳过；这里保持一致即可。
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    trainer.project_name=verl \
    trainer.experiment_name=$RUN_NAME \
    trainer.default_local_dir=$DIR_TO_SAVE_CKPTS \
    trainer.logger=$TRAINER_LOGGER \
    +trainer.val_before_train=False \
    # 你当前是单机 8 卡
    trainer.n_gpus_per_node=8 \
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