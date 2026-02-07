import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import JointState, Imu
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
# dynamics_libはM,C,g,J関数を提供；hardware_apiはアクチュエータ／センサを抽象化
from dynamics_lib import compute_M_C_g, compute_Jacobian
from hardware_api import read_joint_states, write_joint_torques, read_imu

NUM_JOINTS = 7  # 実機に合わせて変更

class ImpedanceController(Node):
    def __init__(self):
        super().__init__('impedance_controller')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('kp', [50.0]*NUM_JOINTS),
                ('kd', [2.0]*NUM_JOINTS),
                ('max_torque', 40.0),
                ('control_freq', 500.0),
            ])
        self.Kp = np.array(self.get_parameter('kp').value, dtype=float)
        self.Kd = np.array(self.get_parameter('kd').value, dtype=float)
        self.max_torque = float(self.get_parameter('max_torque').value)
        period = 1.0 / float(self.get_parameter('control_freq').value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1)

        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray, '/joint_torque_command', qos)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, qos)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, qos)
        self.traj_sub = self.create_subscription(
            JointTrajectoryPoint, '/desired_trajectory', self.traj_callback, qos)

        self.timer = self.create_timer(period, self.control_loop)

        self.q = np.zeros(NUM_JOINTS)
        self.q_dot = np.zeros(NUM_JOINTS)
        self.imu = Imu()
        self.q_des = np.zeros(NUM_JOINTS)
        self.qd_des = np.zeros(NUM_JOINTS)
        self.qdd_des = np.zeros(NUM_JOINTS)

    def joint_callback(self, msg: JointState):
        self.q = np.array(msg.position[:NUM_JOINTS])
        self.q_dot = np.array(msg.velocity[:NUM_JOINTS])

    def imu_callback(self, msg: Imu):
        self.imu = msg

    def traj_callback(self, msg: JointTrajectoryPoint):
        self.q_des = np.array(msg.positions[:NUM_JOINTS])
        self.qd_des = np.array(msg.velocities[:NUM_JOINTS])
        self.qdd_des = np.array(msg.accelerations[:NUM_JOINTS])

    def control_loop(self):
        if self.q.size != NUM_JOINTS or self.q_dot.size != NUM_JOINTS:
            return  # 初期化待ち

        M, C, g = compute_M_C_g(self.q, self.q_dot)
        e = self.q - self.q_des
        ed = self.q_dot - self.qd_des
        tau_ff = M @ self.qdd_des + C @ self.qd_des + g
        tau_fb = - (self.Kp * e + self.Kd * ed)
        tau = tau_ff + tau_fb
        tau = np.clip(tau, -self.max_torque, self.max_torque)

        cmd = Float64MultiArray()
        cmd.data = tau.tolist()
        self.joint_cmd_pub.publish(cmd)
        write_joint_torques(tau)

def main(args=None):
    rclpy.init(args=args)
    node = ImpedanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()