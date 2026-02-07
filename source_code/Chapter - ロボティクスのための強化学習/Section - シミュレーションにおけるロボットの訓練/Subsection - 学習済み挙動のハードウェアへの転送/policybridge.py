#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import torch
import numpy as np
from typing import List, Optional

class PolicyRunner(Node):
    def __init__(self) -> None:
        super().__init__('policy_runner')

        # QoS設定：リアルタイム制御に適したプロファイル
        ctrl_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self._pub = self.create_publisher(
            Float64MultiArray, '/cmd_torque', ctrl_qos
        )

        self._sub = self.create_subscription(
            JointState, '/joint_states', self.joint_cb, ctrl_qos
        )

        # ポリシ読み込み
        self.policy: torch.jit.ScriptModule = torch.jit.load(
            'policy.pt', map_location='cpu'
        )
        self.policy.eval()

        # パラメータ
        self.declare_parameter('num_joints', 7)
        self.declare_parameter('watchdog_timeout', 0.5)
        self.declare_parameter('torque_limits', [-30.0, 30.0])
        self.declare_parameter('delta_max', 5.0)

        self.num_joints: int = self.get_parameter('num_joints').value
        self.watchdog_timeout: float = self.get_parameter('watchdog_timeout').value
        self.tau_min: float = self.get_parameter('torque_limits').value[0]
        self.tau_max: float = self.get_parameter('torque_limits').value[1]
        self.delta_max: float = self.get_parameter('delta_max').value

        self.prev_tau: np.ndarray = np.zeros(self.num_joints)
        self.last_time: float = self.get_clock().now().nanoseconds * 1e-9

        # ウォッチドッグタイマー
        self._watchdog = self.create_timer(0.02, self.watchdog_cb)  # 50 Hz

    def joint_cb(self, msg: JointState) -> None:
        self.last_time = self.get_clock().now().nanoseconds * 1e-9

        # 状態テンソル構築
        q = np.array(msg.position, dtype=np.float32)
        dq = np.array(msg.velocity, dtype=np.float32)
        state = torch.from_numpy(np.concatenate([q, dq])).unsqueeze(0)

        with torch.no_grad():
            a = self.policy(state).squeeze(0).numpy()  # [-1, 1]

        # トルク指令算出 & クリップ
        delta = np.clip(a * self.delta_max, -self.delta_max, self.delta_max)
        tau = np.clip(self.prev_tau + delta, self.tau_min, self.tau_max)
        self.prev_tau = tau

        out = Float64MultiArray()
        out.data = tau.tolist()
        self._pub.publish(out)

    def watchdog_cb(self) -> None:
        if self.get_clock().now().nanoseconds * 1e-9 - self.last_time > self.watchdog_timeout:
            # 安全停止
            stop_msg = Float64MultiArray()
            stop_msg.data = [0.0] * self.num_joints
            self._pub.publish(stop_msg)

def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = PolicyRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()