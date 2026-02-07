import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Int8, Float32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from dataclasses import dataclass
import threading
import time

@dataclass
class ActionCandidate:
    action: np.ndarray
    source: str
    utility: float
    valid: bool

class HybridArbiterNode(Node):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__('hybrid_arbiter', namespace=params.get('namespace', ''))
        self.declare_parameters(
            namespace='',
            parameters=[
                ('weights.bt', 1.0),
                ('weights.rl', 0.9),
                ('weights.mpc', 1.2),
                ('lambda_switch', 0.5),
                ('timeout_sec', 0.1),
                ('rate_hz', 50),
            ]
        )
        self.weights: Dict[str, float] = {
            'bt': self.get_parameter('weights.bt').value,
            'rl': self.get_parameter('weights.rl').value,
            'mpc': self.get_parameter('weights.mpc').value,
        }
        self.lambda_switch: float = self.get_parameter('lambda_switch').value
        self.timeout_sec: float = self.get_parameter('timeout_sec').value
        self.rate_hz: int = self.get_parameter('rate_hz').value

        self.prev_action: Optional[np.ndarray] = None
        self.state_lock = threading.Lock()
        self.latest_state: Optional[Dict[str, np.ndarray]] = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_out', qos)
        self.state_sub = self.create_subscription(
            Odometry, 'odom_in', self._state_cb, qos)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

    def _state_cb(self, msg: Odometry) -> None:
        with self.state_lock:
            self.latest_state = {
                'pose': np.array([
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ]),
                'twist': np.array([
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z,
                    msg.twist.twist.angular.x,
                    msg.twist.twist.angular.y,
                    msg.twist.twist.angular.z,
                ]),
            }

    def query_bt_leaf(self, state: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float, bool]:
        # 実装は実際のBTノードに置き換える
        return np.array([0.0, 0.0]), 0.8, True

    def query_rl_policy(self, state: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float, bool]:
        # 実装は実際のRLポリシに置き換える
        return np.array([0.1, 0.0]), 0.7, True

    def query_mpc(self, state: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float, bool]:
        # 実装は実際のMPCに置き換える
        return np.array([0.05, 0.0]), 0.9, True

    def safety_check(self, state: Dict[str, np.ndarray], action: np.ndarray) -> bool:
        # 簡易衝突判定
        return np.linalg.norm(action) < 2.0

    def tick(self) -> None:
        with self.state_lock:
            if self.latest_state is None:
                return
            state = self.latest_state.copy()

        candidates: List[ActionCandidate] = []
        a_bt, u_bt, v_bt = self.query_bt_leaf(state)
        if v_bt:
            candidates.append(ActionCandidate(a_bt, 'bt', u_bt, v_bt))
        a_rl, u_rl, v_rl = self.query_rl_policy(state)
        if v_rl:
            candidates.append(ActionCandidate(a_rl, 'rl', u_rl, v_rl))
        a_mpc, u_mpc, v_mpc = self.query_mpc(state)
        if v_mpc:
            candidates.append(ActionCandidate(a_mpc, 'mpc', u_mpc, v_mpc))

        safe = [c for c in candidates if self.safety_check(state, c.action)]
        if not safe:
            self.get_logger().warn('No safe action')
            return

        def score(c: ActionCandidate) -> float:
            w = self.weights[c.source]
            switch = 0.0 if np.array_equal(self.prev_action, c.action) else 1.0
            return w * c.utility - self.lambda_switch * switch

        chosen = max(safe, key=score)
        self.prev_action = chosen.action

        twist = Twist()
        twist.linear.x = float(chosen.action[0])
        twist.angular.z = float(chosen.action[1])
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = HybridArbiterNode({})
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()