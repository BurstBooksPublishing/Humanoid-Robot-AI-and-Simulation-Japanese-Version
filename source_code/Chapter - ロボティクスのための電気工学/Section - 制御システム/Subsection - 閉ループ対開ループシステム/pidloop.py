#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
import threading
import numpy as np
from typing import Optional

class JointPidController(Node):
    def __init__(self) -> None:
        super().__init__('joint_pid_controller')

        # --- ROS 2 パラメータ ---
        self.declare_parameter('joint_name', 'joint_1')
        self.declare_parameter('kp', 50.0)
        self.declare_parameter('ki', 10.0)
        self.declare_parameter('kd', 0.5)
        self.declare_parameter('dt', 0.005)          # 200 Hz
        self.declare_parameter('max_torque', 10.0)
        self.declare_parameter('anti_windup', 0.1)   # 積分飽和防止係数

        self.joint_name: str = self.get_parameter('joint_name').value
        self.kp: float = self.get_parameter('kp').value
        self.ki: float = self.get_parameter('ki').value
        self.kd: float = self.get_parameter('kd').value
        self.dt: float = self.get_parameter('dt').value
        self.max_torque: float = self.get_parameter('max_torque').value
        self.anti_windup: float = self.get_parameter('anti_windup').value

        # --- 状態変数 ---
        self.q_ref: float = 0.0
        self.q: float = 0.0
        self.qdot: float = 0.0
        self.error: float = 0.0
        self.error_prev: float = 0.0
        self.integrator: float = 0.0

        # --- 同期用 ---
        self.state_lock = threading.Lock()

        # --- QoS ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- 購読 ---
        self.sub_js = self.create_subscription(
            JointState, 'joint_states', self.cb_js, qos)
        self.sub_ref = self.create_subscription(
            JointTrajectoryPoint, 'joint_reference', self.cb_ref, qos)

        # --- 配信 ---
        self.pub_eff = self.create_publisher(Float64, 'effort_command', qos)

        # --- タイマー ---
        self.timer = self.create_timer(self.dt, self.control_cycle)

        self.get_logger().info(f'{self.joint_name} PID controller ready')

    # -- コールバック --
    def cb_js(self, msg: JointState) -> None:
        idx = msg.name.index(self.joint_name) if self.joint_name in msg.name else -1
        if idx < 0:
            return
        with self.state_lock:
            self.q = msg.position[idx]
            self.qdot = msg.velocity[idx] if len(msg.velocity) > idx else 0.0

    def cb_ref(self, msg: JointTrajectoryPoint) -> None:
        with self.state_lock:
            self.q_ref = msg.positions[0] if msg.positions else self.q_ref

    # -- 制御周期 --
    def control_cycle(self) -> None:
        with self.state_lock:
            q_ref = self.q_ref
            q = self.q
            qdot = self.qdot

        error = q_ref - q
        self.integrator += error * self.dt

        # 積分飽和防止
        self.integrator = np.clip(
            self.integrator,
            -self.anti_windup / self.ki,
            self.anti_windup / self.ki
        )

        derivative = (error - self.error_prev) / self.dt
        tau_ff = 0.0  # 必要に応じてfeedforwardを追加
        tau = self.kp * error + self.ki * self.integrator + self.kd * derivative + tau_ff
        tau = float(np.clip(tau, -self.max_torque, self.max_torque))

        self.error_prev = error

        # 指令送信
        cmd = Float64()
        cmd.data = tau
        self.pub_eff.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = JointPidController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()