import numpy as np
import torch
import gymnasium as gym
from typing import Protocol, Tuple, Any


class PolicyProtocol(Protocol):
    """Policy must expose deterministic act()."""
    def act(self, state: np.ndarray) -> np.ndarray: ...


class ZMPModelProtocol(Protocol):
    """ZMP predictor must expose predict() and is_within_support()."""
    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray: ...
    def is_within_support(self, zmp: np.ndarray) -> bool: ...


class EpsilonGreedySafe:
    """
    ε-greedy 探索戦略に安全射影を組み込んだ行動選択クラス
    """

    def __init__(
        self,
        policy: PolicyProtocol,
        env: gym.Env,
        eps_start: float = 0.5,
        eps_min: float = 0.05,
        decay: float = 1e-4,
        zmp_model: ZMPModelProtocol | None = None,
        torque_rate_limit: float | None = None,
    ) -> None:
        self.policy = policy
        self.env = env
        self.eps = eps_start
        self.eps_min = eps_min
        self.decay = decay
        self.zmp_model = zmp_model
        self.torque_rate_limit = torque_rate_limit
        self._last_action: np.ndarray | None = None

    def reset(self) -> None:
        """エピソード開始時に内部状態をリセット"""
        self._last_action = None

    def step(self, state: np.ndarray, step_count: int) -> np.ndarray:
        # ε を指数減衰
        self.eps = max(self.eps_min, self.eps * (1.0 - self.decay))

        if np.random.rand() < self.eps:
            action = self.env.action_space.sample()  # 安全なランダム探索
        else:
            action = self.policy.act(state)

        return self._safety_projection(action, state)

    def _safety_projection(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        # 関節限界クリップ
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = np.clip(action, low, high)

        # トルク変化率制限
        if self.torque_rate_limit is not None and self._last_action is not None:
            delta = np.clip(
                action - self._last_action,
                -self.torque_rate_limit,
                self.torque_rate_limit,
            )
            action = self._last_action + delta

        # ZMP 安定性チェック
        if self.zmp_model is not None and self._predict_instability(state, action):
            action *= 0.5  # 保守的にスケールダウン

        self._last_action = action.copy()
        return action

    def _predict_instability(self, state: np.ndarray, action: np.ndarray) -> bool:
        zmp = self.zmp_model.predict(state, action)
        return not self.zmp_model.is_within_support(zmp)