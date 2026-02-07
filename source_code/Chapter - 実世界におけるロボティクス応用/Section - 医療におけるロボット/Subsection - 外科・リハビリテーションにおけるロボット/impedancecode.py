#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from builtin_interfaces.msg import Time
from geometry_msgs.msg import WrenchStamped, TwistStamped, PoseStamped
from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import JointState
import threading
import time

class CartesianImpedanceController(Node):
    def __init__(self):
        super().__init__('cartesian_impedance_controller')

        # パラメータ
        self.declare_parameter('dt', 0.001)
        self.declare_parameter('Kd_diag', [2000.0]*3 + [50.0]*3)
        self.declare_parameter('Bd_diag', [50.0]*3 + [5.0]*3)
        self.declare_parameter('Md_diag', [1.0]*3 + [0.1]*3)
        self.declare_parameter('force_limit', 30.0)

        self.dt = self.get_parameter('dt').value
        self.Kd = np.diag(self.get_parameter('Kd_diag').value)
        self.Bd = np.diag(self.get_parameter('Bd_diag').value)
        self.Md = np.diag(self.get_parameter('Md_diag').value)
        self.force_limit = self.get_parameter('force_limit').value

        # 購読
        self.create_subscription(PoseStamped, 'current_pose', self.cb_pose, 1)
        self.create_subscription(TwistStamped, 'current_twist', self.cb_twist, 1)
        self.create_subscription(WrenchStamped, 'external_wrench', self.cb_wrench, 1)
        self.create_subscription(PoseStamped, 'desired_pose', self.cb_des_pose, 1)
        self.create_subscription(TwistStamped, 'desired_twist', self.cb_des_twist, 1)
        self.create_subscription(JointState, 'joint_states', self.cb_joints, 1)

        # 配信
        self.tau_pub = self.create_publisher(Float64MultiArray, 'joint_torque_command', 1)
        self.stop_pub = self.create_publisher(Bool, 'emergency_stop', 1)

        # 状態変数
        self.x = np.zeros(6)
        self.xd = np.zeros(6)
        self.x_dot = np.zeros(6)
        self.xd_dot = np.zeros(6)
        self.Fext = np.zeros(6)
        self.q = np.array([])
        self.J = np.zeros((6, 7))  # 7DoF仮定

        self.lock = threading.Lock()
        self.timer = self.create_timer(self.dt, self.control_loop)

    def cb_pose(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        # 簡易的に位置+ZYXオイラー角に変換
        with self.lock:
            self.x[:3] = [p.x, p.y, p.z]
            # オイラー角変換省略（実機では適切に変換）

    def cb_twist(self, msg: TwistStamped):
        with self.lock:
            self.x_dot[0] = msg.twist.linear.x
            self.x_dot[1] = msg.twist.linear.y
            self.x_dot[2] = msg.twist.linear.z
            self.x_dot[3] = msg.twist.angular.x
            self.x_dot[4] = msg.twist.angular.y
            self.x_dot[5] = msg.twist.angular.z

    def cb_des_pose(self, msg: PoseStamped):
        p = msg.pose.position
        with self.lock:
            self.xd[:3] = [p.x, p.y, p.z]

    def cb_des_twist(self, msg: TwistStamped):
        with self.lock:
            self.xd_dot[0] = msg.twist.linear.x
            self.xd_dot[1] = msg.twist.linear.y
            self.xd_dot[2] = msg.twist.linear.z
            self.xd_dot[3] = msg.twist.angular.x
            self.xd_dot[4] = msg.twist.angular.y
            self.xd_dot[5] = msg.twist.angular.z

    def cb_wrench(self, msg: WrenchStamped):
        with self.lock:
            self.Fext[0] = msg.wrench.force.x
            self.Fext[1] = msg.wrench.force.y
            self.Fext[2] = msg.wrench.force.z
            self.Fext[3] = msg.wrench.torque.x
            self.Fext[4] = msg.wrench.torque.y
            self.Fext[5] = msg.wrench.torque.z

    def cb_joints(self, msg: JointState):
        with self.lock:
            self.q = np.array(msg.position)

    def compute_jacobian(self):
        # 実機ではFK/Jacobianライブラリ使用
        return np.eye(6, len(self.q))

    def compute_nullspace_torque(self):
        # 簡易ポスチャ制御（重力補償含む）
        return np.zeros(len(self.q))

    def control_loop(self):
        with self.lock:
            x = self.x.copy()
            xd = self.xd.copy()
            x_dot = self.x_dot.copy()
            xd_dot = self.xd_dot.copy()
            Fext = self.Fext.copy()
            J = self.compute_jacobian()

        e = x - xd
        edot = xd_dot - x_dot
        # インピーダンス則
        Fdes = self.Md @ (-edot / self.dt) + self.Bd @ edot + self.Kd @ e

        # 安全チェック
        if np.linalg.norm(Fext) > self.force_limit:
            Fdes = np.zeros(6)
            self.stop_pub.publish(Bool(data=True))

        tau = J.T @ Fdes + self.compute_nullspace_torque()
        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.tau_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CartesianImpedanceController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()