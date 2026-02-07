import numpy as np
from typing import Tuple, Dict, Any
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.world import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage


class HumanoidEnv:
    """Isaac-Sim上で人型アシムートをPD制御するRL環境."""

    def __init__(
        self,
        usd_path: str,
        dt: float = 1 / 60,
        device: str = "cuda:0",
    ) -> None:
        self._world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
        self._device = device
        self.dt = dt

        # USDをステージに追加
        add_reference_to_stage(usd_path, "/World/Humanoid")
        # ArticulationViewで並列シミュレーション対応
        self.humanoid = ArticulationView(
            prim_paths_expr="/World/Humanoid.*", name="humanoid_view"
        )
        self._world.scene.add(self.humanoid)

        # 関節数の取得
        self.num_dof = self.humanoid.num_dof
        assert self.num_dof > 0, "有効な関節が見つかりません"

        # PDゲイン
        self.Kp = np.full(self.num_dof, 100.0, dtype=np.float32)
        self.Kd = np.full(self.num_dof, 2.0, dtype=np.float32)

        # 初期姿勢
        self.default_q = np.zeros(self.num_dof, dtype=np.float32)

        # シミュレーション待機
        self._world.reset()

    def reset(self) -> np.ndarray:
        """環境を初期状態に戻す."""
        self._world.reset()

        # ルート位置・姿勢を初期化
        root_pos = np.array([0.0, 0.0, 0.9], dtype=np.float32)
        root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.humanoid.set_world_poses(root_pos.reshape(1, 3), root_quat.reshape(1, 4))

        # 関節位置・速度を初期化
        self.humanoid.set_joint_positions(self.default_q)
        self.humanoid.set_joint_velocities(np.zeros(self.num_dof, dtype=np.float32))

        # 1ステップ進めて物理を安定
        self._world.step(render=False)
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """1ステップ進める."""
        action = np.clip(action, -np.pi, np.pi).astype(np.float32)

        # 現在状態取得
        q = self.humanoid.get_joint_positions()
        qd = self.humanoid.get_joint_velocities()

        # PDトルク計算
        tau = self.Kp * (action - q) - self.Kd * qd
        self.humanoid.set_joint_efforts(tau)

        # シミュレーション進行
        self._world.step(render=True)

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self._check_termination(obs)
        return obs, reward, done, {}

    def _get_obs(self) -> np.ndarray:
        """観測を返す."""
        q = self.humanoid.get_joint_positions()
        qd = self.humanoid.get_joint_velocities()
        return np.concatenate([q, qd]).astype(np.float32)

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """報酬計算."""
        # 起立報酬（腰の高さ）
        base_z = self.humanoid.get_world_poses()[0][0, 2]
        upright_reward = base_z

        # エネルギー消費ペナルティ
        energy_penalty = 1e-3 * np.sum(np.square(action))

        return upright_reward - energy_penalty

    def _check_termination(self, obs: np.ndarray) -> bool:
        """転倒判定."""
        base_z = self.humanoid.get_world_poses()[0][0, 2]
        return base_z < 0.3  # 腰が0.3 m未満で終了