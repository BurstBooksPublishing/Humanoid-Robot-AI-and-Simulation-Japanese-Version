import numpy as np
from typing import Dict, Any, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class RewardCalculator(Node):
    """
    ROS 2ノードとして報酬計算を提供。
    トピック経由で観測・行動を受信し、報酬を返す。
    """

    def __init__(self, node_name: str = "reward_calculator") -> None:
        super().__init__(node_name)

        # パラメータ宣言 & 取得
        self.declare_parameters(
            namespace="",
            parameters=[
                ("v_target", [1.0, 0.0, 0.0]),
                ("alpha", 1.0),
                ("beta", 1e-4),
                ("gamma", 1e-3),
                ("foot_bonus", 5.0),
                ("discount", 0.99),
                ("forward_axis", [1.0, 0.0, 0.0]),
                ("w_upright", 1.0),
                ("w_com", 1.0),
                ("w_energy", 1.0),
                ("w_contact", 1.0),
                ("w_foot", 1.0),
                ("w_shaping", 1.0),
                ("fall_penalty", 100.0),
                ("reward_clip", 10.0),
            ],
        )
        self.params: Dict[str, Any] = {
            key: self.get_parameter(key).value for key in (
                "v_target", "alpha", "beta", "gamma", "foot_bonus", "discount",
                "forward_axis", "w_upright", "w_com", "w_energy", "w_contact",
                "w_foot", "w_shaping", "fall_penalty", "reward_clip"
            )
        }
        self.v_target = np.asarray(self.params["v_target"], dtype=np.float32)
        self.forward_axis = np.asarray(self.params["forward_axis"], dtype=np.float32)

        # 実行統計の初期化
        self.running_stats: Dict[str, float] = {
            "upright_mean": 0.0,
            "upright_std": 1.0,
            "com_scale": 1.0,
        }

        # 購読・配信
        self.obs_sub = self.create_subscription(
            Float32MultiArray, "/observation", self._obs_callback, 10
        )
        self.action_sub = self.create_subscription(
            Float32MultiArray, "/action", self._action_callback, 10
        )
        self.reward_pub = self.create_publisher(Float32MultiArray, "/reward", 10)

        self._latest_obs: Optional[Dict[str, np.ndarray]] = None
        self._latest_action: Optional[np.ndarray] = None

    def _obs_callback(self, msg: Float32MultiArray) -> None:
        # メッセージを辞書にデコード
        obs = self._decode_obs(msg)
        self._latest_obs = obs
        self._try_compute()

    def _action_callback(self, msg: Float32MultiArray) -> None:
        self._latest_action = np.asarray(msg.data, dtype=np.float32)
        self._try_compute()

    def _try_compute(self) -> None:
        if self._latest_obs is None or self._latest_action is None:
            return
        reward = self.compute_reward(
            self._latest_obs, self._latest_action, self.running_stats, self.params
        )
        out_msg = Float32MultiArray()
        out_msg.data = [reward]
        self.reward_pub.publish(out_msg)

    @staticmethod
    def _decode_obs(msg: Float32MultiArray) -> Dict[str, np.ndarray]:
        # 簡易デコーダ：実際のメッセージ定義に合わせて調整
        arr = np.asarray(msg.data, dtype=np.float32)
        return {
            "torso_z": arr[0:3],
            "com_vel": arr[3:6],
            "com_pos": arr[6:9],
            "contact_impulses": arr[9:13],
            "correct_foot_phase": arr[13],
            "fell": bool(arr[14]),
        }

    def compute_reward(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        running_stats: Dict[str, float],
        params: Dict[str, Any],
    ) -> float:
        """
        観測と行動から報酬を計算。
        """
        # 起立性報酬
        upright = max(0.0, float(np.dot(obs["torso_z"], np.array([0.0, 0.0, 1.0]))))

        # 重心速度追従
        v_err = obs["com_vel"] - self.v_target
        com_term = -params["alpha"] * float(np.dot(v_err, v_err))

        # エネルギー消費ペナルティ
        energy_term = -params["beta"] * float(np.sum(np.square(action)))

        # 接触衝撃ペナルティ
        contact_term = -params["gamma"] * float(np.sum(np.square(obs["contact_impulses"])))

        # 足接地位相報酬
        foot_reward = params["foot_bonus"] * float(obs["correct_foot_phase"])

        # ポテンシャルベース shaping
        phi_s = float(np.dot(obs["com_pos"], self.forward_axis))
        # 次状態が無いため同一フレームで近似（実環境では次状態を受け取る）
        shaping = (params["discount"] - 1.0) * phi_s

        # 正規化
        upright_n = (upright - running_stats["upright_mean"]) / (
            running_stats["upright_std"] + 1e-8
        )
        com_n = com_term / (running_stats["com_scale"] + 1e-8)

        # 重み付き合計
        reward = (
            params["w_upright"] * upright_n
            + params["w_com"] * com_n
            + params["w_energy"] * energy_term
            + params["w_contact"] * contact_term
            + params["w_foot"] * foot_reward
            + params["w_shaping"] * shaping
        )

        # 転倒ペナルティ
        if obs["fell"]:
            reward -= params["fall_penalty"]

        # クリップ
        return float(np.clip(reward, -params["reward_clip"], params["reward_clip"]))


def main(args=None):
    rclpy.init(args=args)
    node = RewardCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()