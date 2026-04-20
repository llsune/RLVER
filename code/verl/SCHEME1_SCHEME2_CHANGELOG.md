# Scheme-1 & Scheme-2 Code Change Log

This document records the staged implementation for multi-turn RL improvements:
- Scheme-1: Emit trajectory metadata from multi-turn rollout flattening.
- Scheme-2: Add optional trajectory-aware turn-credit reward mixing in URL environment.

All code changes are limited to files under code/verl.

## 1) Modified Files

### 1. workers/rollout/vllm_rollout/vllm_rollout_spmd_think.py

#### What changed
- Added rollout-flattened metadata arrays:
  - trajectory_id
  - turn_idx
  - trajectory_len
- Exported the above fields in non_tensor_batch together with existing step_reward.

#### Why
- Needed by downstream trajectory-aware reward shaping and state grouping.
- Keeps trainer/environment logic simple by carrying explicit row-level trajectory semantics.

#### Marker comments added
- === RLVER PATCH [SCHEME-1]: Track per-row trajectory metadata after flattening ===
- === RLVER PATCH [SCHEME-1]: Row-level identifiers for trajectory-aware reward shaping/grouping ===
- === RLVER PATCH [SCHEME-1]: Expose trajectory metadata for cross-turn return mixing ===

#### Behavior impact
- No change to existing tensor batch keys (prompts, responses, input_ids, etc.).
- Adds extra non-tensor columns only.

---

### 2. environments/url_environment.py

#### What changed
- Added helper method:
  - _get_turn_credit_config()
- Added optional turn-credit reward mixing path in get_reward_batched:
  - Reads trajectory_id and turn_idx when enabled.
  - Computes per-trajectory reverse discounted return.
  - Blends immediate and return reward:
    - mixed = (1 - alpha) * step + alpha * return

#### Why
- Improve cross-turn credit assignment in multi-turn environments while preserving existing sparse token write behavior.

#### Marker comments added
- === RLVER PATCH [SCHEME-2]: Centralized turn-credit config to keep reward logic deterministic ===
- === RLVER PATCH [SCHEME-2]: Blend immediate step reward with trajectory-level turn return ===
- === RLVER PATCH [SCHEME-2]: Keep raw step reward for observability/metrics ===

#### Behavior impact
- Default behavior unchanged (turn_credit.enable: False).
- When enabled:
  - penalized_reward_tensor uses mixed rewards.
  - original_reward_tensor remains raw step_reward for observability.

#### Fallback behavior
- If turn_credit enabled but trajectory_id/turn_idx missing, code falls back to raw step_reward.

---

### 3. trainer/config/ppo_trainer.yaml

#### What changed
- Added new algorithm config block:

algorithm:
  turn_credit:
    enable: False
    alpha: 0.4
    gamma: 0.97

#### Why
- Make Scheme-2 behavior configurable and backward compatible.

#### Marker comments added
- === RLVER PATCH [SCHEME-2]: Cross-turn return mixing for multi-turn environments ===

## 2) Operational Notes

### Enable Scheme-2 in runtime overrides
Set for experiment runs:
- algorithm.turn_credit.enable=True
- Optional tuning:
  - algorithm.turn_credit.alpha=0.4
  - algorithm.turn_credit.gamma=0.97

### Recommended initial sweep
- Alpha: 0.2, 0.4, 0.6
- Gamma: 0.95, 0.97, 0.99

## 3) Compatibility & Safety

- Backward compatible by default (enable=False).
- No changes made to trainer-side PPO update equations.
- No changes made outside code/verl.

## 4) Validation Performed

Static diagnostics after edits:
- workers/rollout/vllm_rollout/vllm_rollout_spmd_think.py: no errors
- environments/url_environment.py: no errors
- trainer/config/ppo_trainer.yaml: no errors

## 5) Rollback Guide

To disable new behavior immediately:
- Keep algorithm.turn_credit.enable=False (default)

To fully revert code-level changes:
- Revert only the three modified files listed above.
