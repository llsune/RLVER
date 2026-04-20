from verl import DataProto
import requests
import torch
import numpy as np


class URLEnvironment():

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    # === RLVER PATCH [SCHEME-2]: Centralized turn-credit config to keep reward logic deterministic ===
    def _get_turn_credit_config(self):
        algorithm_cfg = self.config.get('algorithm', {})
        turn_credit_cfg = algorithm_cfg.get('turn_credit', {})
        enable = bool(turn_credit_cfg.get('enable', False))
        alpha = float(turn_credit_cfg.get('alpha', 0.4))
        gamma = float(turn_credit_cfg.get('gamma', 0.97))
        alpha = min(max(alpha, 0.0), 1.0)
        gamma = min(max(gamma, 0.0), 1.0)
        return enable, alpha, gamma


    def get_reward_batched(self, data: DataProto):  #batched
        messages_batched = []
        reward_locs = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            messages = data_item.non_tensor_batch['messages']
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            messages_batched.append(messages)

            attention_mask = data_item.batch['attention_mask']
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = attention_mask[prompt_length:].sum()
            reward_locs.append(valid_response_length - 1)

        # [Added/Modified for Dense Reward]: 直接读取 rollout 已计算好的逐轮密集奖励（可正可负）。
        reward_batched = np.asarray(data.non_tensor_batch['step_reward'], dtype=np.float32)
        # === RLVER PATCH [SCHEME-2]: Keep raw step reward for observability/metrics ===
        original_reward_batched = reward_batched.copy()

        # === RLVER PATCH [SCHEME-2]: Blend immediate step reward with trajectory-level turn return ===
        enable_turn_credit, turn_credit_alpha, turn_credit_gamma = self._get_turn_credit_config()
        if enable_turn_credit and 'trajectory_id' in data.non_tensor_batch and 'turn_idx' in data.non_tensor_batch:
            trajectory_ids = np.asarray(data.non_tensor_batch['trajectory_id'], dtype=np.int64)
            turn_indices = np.asarray(data.non_tensor_batch['turn_idx'], dtype=np.int64)
            mixed_reward_batched = reward_batched.copy()

            for traj_id in np.unique(trajectory_ids):
                traj_mask = trajectory_ids == traj_id
                traj_rows = np.where(traj_mask)[0]
                traj_order = np.argsort(turn_indices[traj_rows])
                traj_rows_sorted = traj_rows[traj_order]

                running_return = 0.0
                for row_idx in reversed(traj_rows_sorted):
                    step_reward = float(reward_batched[row_idx])
                    running_return = step_reward + turn_credit_gamma * running_return
                    mixed_reward_batched[row_idx] = (
                        (1.0 - turn_credit_alpha) * step_reward + turn_credit_alpha * running_return
                    )

            reward_batched = mixed_reward_batched

        # [Added/Modified for Dense Reward]: 保持负奖励以惩罚不当回复，不做截断或归一化。
        # reward_batched = data.non_tensor_batch['emo_point']/100
        # reward_batched = np.maximum(reward_batched, 0)
        # original_reward_batched = reward_batched.copy()



        original_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        penalized_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        for i in range(len(data)):
            original_reward_tensor[i, reward_locs[i]] = original_reward_batched[i]
            penalized_reward_tensor[i, reward_locs[i]] = reward_batched[i]
        
        return original_reward_tensor, penalized_reward_tensor