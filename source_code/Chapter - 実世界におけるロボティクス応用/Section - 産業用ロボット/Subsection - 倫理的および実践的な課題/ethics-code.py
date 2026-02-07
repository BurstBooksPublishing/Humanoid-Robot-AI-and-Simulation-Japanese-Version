import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from typing import NamedTuple


class State(NamedTuple):
    p_robot: np.ndarray
    p_human: np.ndarray
    taus: np.ndarray
    p_fail: float
    severity: float


class SafetyMonitor(Node):
    def __init__(self,
                 d_min: float,
                 tau_max: float,
                 risk_threshold: float,
                 node_name: str = "safety_monitor"):
        super().__init__(node_name)
        self._d_min = d_min
        self._tau_max = tau_max
        self._risk_threshold = risk_threshold

        # 緊急停止指令パブリッシャ
        self._estop_pub = self.create_publisher(Bool, "/safety/emergency_stop", 1)

        # ログ
        self.get_logger().info(
            f"d_min={d_min:.3f}, tau_max={tau_max:.3f}, risk_threshold={risk_threshold:.3f}"
        )

    def compute_risk(self,
                     p_robot: np.ndarray,
                     p_human: np.ndarray,
                     taus: np.ndarray,
                     p_fail: float,
                     severity: float) -> float:
        d = np.linalg.norm(p_robot - p_human)
        # 距離要因：d_min未満なら危険度上昇
        dist_factor = max(0.0, (self._d_min - d) / self._d_min)
        # トルク要因：正規化平均
        torque_factor = np.mean(np.abs(taus) / self._tau_max)
        # 統合リスク
        risk = p_fail * severity * (1.0 + dist_factor + torque_factor)
        return float(risk)

    def check_and_act(self, state: State) -> None:
        risk = self.compute_risk(
            state.p_robot,
            state.p_human,
            state.taus,
            state.p_fail,
            state.severity
        )
        if risk > self._risk_threshold:
            self.emergency_stop()
        else:
            self.allow_motion()

    def emergency_stop(self) -> None:
        # 緊急停止フラグを送信
        msg = Bool()
        msg.data = True
        self._estop_pub.publish(msg)
        self.get_logger().error("Emergency stop triggered!")

    def allow_motion(self) -> None:
        # 通常運転継続（何もしない）
        pass