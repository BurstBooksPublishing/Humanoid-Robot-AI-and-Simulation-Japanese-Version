import numpy as np
from typing import Sequence, Union, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool


class StateEstimator(Node):
    """
    ROS 2ノードとして動作する状態推定器。
    シミュレータ／実機のセンサ値を購読し、正規化された状態ベクトルを返す。
    """

    def __init__(self, joint_names: Sequence[str], foot_names: Sequence[str],
                 nominal_height: float = 1.0,
                 node_name: str = "state_estimator") -> None:
        super().__init__(node_name)

        self.joint_names = joint_names
        self.foot_names = foot_names
        self.nominal_height = nominal_height

        # 最新センサ値を保持
        self.q: Optional[np.ndarray] = None
        self.qd: Optional[np.ndarray] = None
        self.imu_ori: Optional[np.ndarray] = None
        self.imu_omega: Optional[np.ndarray] = None
        self.com: Optional[np.ndarray] = None
        self.foot_contacts: Optional[np.ndarray] = None

        # 購読
        self.create_subscription(JointState, "joint_states", self._cb_joint, 1)
        self.create_subscription(Imu, "imu/data", self._cb_imu, 1)
        self.create_subscription(Vector3, "com_est", self._cb_com, 1)
        for name in foot_names:
            self.create_subscription(Bool, f"{name}/contact", self._cb_contact(name), 1)

    def _cb_joint(self, msg: JointState) -> None:
        # 関節名順にソートしてから格納
        order = [msg.name.index(n) for n in self.joint_names]
        self.q = np.array([msg.position[i] for i in order], dtype=np.float32)
        self.qd = np.array([msg.velocity[i] for i in order], dtype=np.float32)

    def _cb_imu(self, msg: Imu) -> None:
        o = msg.orientation
        self.imu_ori = np.array([o.x, o.y, o.z, o.w], dtype=np.float32)
        w = msg.angular_velocity
        self.imu_omega = np.array([w.x, w.y, w.z], dtype=np.float32)

    def _cb_com(self, msg: Vector3) -> None:
        self.com = np.array([msg.x, msg.y, msg.z], dtype=np.float32)

    def _cb_contact(self, name: str):
        def cb(msg: Bool) -> None:
            idx = self.foot_names.index(name)
            if self.foot_contacts is None:
                self.foot_contacts = np.zeros(len(self.foot_names), dtype=np.float32)
            self.foot_contacts[idx] = float(msg.data)
        return cb

    def ready(self) -> bool:
        return all(x is not None for x in [self.q, self.qd, self.imu_ori,
                                           self.imu_omega, self.com, self.foot_contacts])

    def get_state(self) -> np.ndarray:
        """
        正規化済み状態ベクトルを返す。ready()==Falseの場合は空配列。
        """
        if not self.ready():
            return np.empty(0, dtype=np.float32)

        com_height = self.com[2]
        zmp_err = 0.0  # 外部ノードで計算済みと仮定
        state = np.concatenate([
            self.q,
            self.qd,
            self.imu_ori,
            self.imu_omega,
            [com_height, zmp_err],
            self.foot_contacts
        ]).astype(np.float32)
        return state


class RewardCalculator:
    """
    状態・行動から報酬を計算。転倒判定はfoot_contact_sum==0。
    """

    def __init__(self, nominal_height: float = 1.0,
                 w_stab: float = 2.0,
                 w_height: float = 1.0,
                 w_energy: float = 0.001,
                 w_fall: float = 100.0) -> None:
        self.nominal_height = nominal_height
        self.w_stab = w_stab
        self.w_height = w_height
        self.w_energy = w_energy
        self.w_fall = w_fall

    def __call__(self, state: np.ndarray, action: np.ndarray,
                 next_state: np.ndarray) -> float:
        n_foot = len(next_state) - 7  # 末尾: [com_height, zmp_err, foot_contacts...]
        foot_contacts = next_state[-n_foot:]
        com_height = next_state[-(n_foot + 2)]
        zmp_err = next_state[-(n_foot + 1)]

        stability = -abs(zmp_err)
        height_pref = -abs(com_height - self.nominal_height)
        energy_penalty = -self.w_energy * np.sum(np.square(action))
        fall_penalty = -self.w_fall if foot_contacts.sum() == 0 else 0.0

        reward = (self.w_stab * stability +
                  self.w_height * height_pref +
                  energy_penalty +
                  fall_penalty)
        return float(reward)