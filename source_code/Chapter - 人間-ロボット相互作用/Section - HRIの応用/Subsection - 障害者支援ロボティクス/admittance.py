import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench, Twist
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import numpy as np


class AdmittanceController(Node):
    def __init__(self) -> None:
        super().__init__('admittance_controller')

        # パラメータ宣言
        self.declare_parameter('mass', 1.0, ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Virtual mass [kg]'))
        self.declare_parameter('damping', 20.0, ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Virtual damping [Ns/m]'))
        self.declare_parameter('stiffness', 50.0, ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Virtual stiffness [N/m]'))
        self.declare_parameter('dt', 0.01, ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Control period [s]'))
        self.declare_parameter('max_vel', 0.2, ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description='Velocity limit [m/s]'))

        # パラメータ取得
        self.M = self.get_parameter('mass').value
        self.D = self.get_parameter('damping').value
        self.K = self.get_parameter('stiffness').value
        self.dt = self.get_parameter('dt').value
        self.max_vel = self.get_parameter('max_vel').value

        # 購読・配信
        self.force_sub = self.create_subscription(
            Wrench, '/force_sensor', self.force_cb, 10)
        self.vel_pub = self.create_publisher(Twist, '/ee_velocity', 10)

        # 内部状態（1D）
        self.x = 0.0
        self.x_dot = 0.0
        self.last_force = 0.0

        # タイマー
        self.timer = self.create_timer(self.dt, self.loop)

    def force_cb(self, msg: Wrench) -> None:
        self.last_force = msg.force.z

    def loop(self) -> None:
        f_ext = self.last_force
        # 半陰的オイラー積分
        x_ddot = (f_ext - self.D * self.x_dot - self.K * self.x) / self.M
        self.x_dot += x_ddot * self.dt
        self.x += self.x_dot * self.dt

        # 速度制限
        self.x_dot = np.clip(self.x_dot, -self.max_vel, self.max_vel)

        # 指令速度配信
        twist = Twist()
        twist.linear.z = self.x_dot
        self.vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = AdmittanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()