#!/usr/bin/env python3
"""
PD+重力補儧＋慣性FF 単関節トルクコマンド発行ノード
ROS 2 (rclpy)対応、パラメータサーバ経由でゲイン・目標値を動的変更可能。
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3Stamped   # theta, theta_dot, theta_ddot_des 受信用
import numpy as np


class SingleJointController(Node):
    def __init__(self):
        super().__init__('single_joint_controller')

        # ---- パラメータ宣言 ----
        self.declare_parameter('Kp', 80.0)
        self.declare_parameter('Kd', 2.0)
        self.declare_parameter('inertia', 0.05)
        self.declare_parameter('gravity_gain', 9.81 * 0.5)

        # ---- 購読 ----
        self.create_subscription(Vector3Stamped, 'joint_state', self.cb_state, 10)
        self.create_subscription(Vector3Stamped, 'target', self.cb_target, 10)

        # ---- 配信 ----
        self.pub_torque = self.create_publisher(Float64, 'torque_cmd', 10)

        # ---- 内部状態 ----
        self.theta = 0.0
        self.theta_dot = 0.0
        self.theta_des = 0.0
        self.theta_dot_des = 0.0
        self.theta_ddot_des = 0.0

        # ---- 制御周期 ----
        self.dt = 0.001  # 1 kHz
        self.timer = self.create_timer(self.dt, self.control_cycle)

    # ---- パラメータ取得ヘルパ ----
    def gain(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    # ---- 重力補償・慣性モデル ----
    def gravity_torque(self, theta: float) -> float:
        return self.gain('gravity_gain') * np.cos(theta)

    def inertia(self, _theta: float) -> float:
        return self.gain('inertia')

    # ---- 状態・目標コールバック ----
    def cb_state(self, msg: Vector3Stamped):
        self.theta = msg.vector.x
        self.theta_dot = msg.vector.y

    def cb_target(self, msg: Vector3Stamped):
        self.theta_des = msg.vector.x
        self.theta_dot_des = msg.vector.y
        self.theta_ddot_des = msg.vector.z

    # ---- 制御周期 ----
    def control_cycle(self):
        Kp = self.gain('Kp')
        Kd = self.gain('Kd')

        # PD + FF + 重力補償
        tau_pd = Kp * (self.theta_des - self.theta) + Kd * (self.theta_dot_des - self.theta_dot)
        tau_ff = self.inertia(self.theta) * self.theta_ddot_des
        tau_grav = self.gravity_torque(self.theta)
        tau = tau_pd + tau_ff + tau_grav

        # 飽和（簡易セーフティ）
        tau = np.clip(tau, -10.0, 10.0)

        # 配信
        msg = Float64()
        msg.data = tau
        self.pub_torque.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SingleJointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()