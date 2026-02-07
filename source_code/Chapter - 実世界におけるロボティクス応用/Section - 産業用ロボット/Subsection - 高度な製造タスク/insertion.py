import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
import numpy as np
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R

# 自作モジュール（同一パッケージ内に配置）
from control_api import ImpedanceController, MPCPlanner, VisionPose


class ImpedanceMPCNode(Node):
    """ROS 2ノード：視覚姿勢を使ったMPC＋インピーダンス制御"""

    def __init__(self) -> None:
        super().__init__('impedance_mpc_node')

        # QoS：リアルタイム優先
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # パブリッシャ／サブスクライバ
        self.cmd_pub = self.create_publisher(JointState, '/robot/joint_cmd', qos)
        self.wrench_sub = self.create_subscription(
            WrenchStamped, '/robot/wrench', self.wrench_cb, qos)
        self.joint_sub = self.create_subscription(
            JointState, '/robot/joint_states', self.joint_cb, qos)

        # 制御器／プランナ
        self.imp = ImpedanceController(
            stiffness=np.diag([800, 800, 800, 50, 50, 50]))
        self.mpc = MPCPlanner(horizon=20, dt=0.02)

        # 内部状態
        self.latest_wrench = np.zeros(6)
        self.latest_joint = JointState()

        # タイマー：500 Hz
        self.timer = self.create_timer(0.002, self.control_loop)

    def wrench_cb(self, msg: WrenchStamped) -> None:
        self.latest_wrench = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])

    def joint_cb(self, msg: JointState) -> None:
        self.latest_joint = msg

    def current_state(self) -> np.ndarray:
        # 簡易：ジョイント位置をMPC状態へ変換
        return np.array(self.latest_joint.position)

    def send_to_hw(self, u: np.ndarray) -> None:
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = self.latest_joint.name
        cmd.effort = u.tolist()
        self.cmd_pub.publish(cmd)

    def log_metrics(self, pose: np.ndarray, u: np.ndarray) -> None:
        self.get_logger().debug(
            f'pose={pose[:3]}, cmd={u[:3]}', throttle_duration_sec=1.0)

    def control_loop(self) -> None:
        pose_cam, cov = VisionPose.get_estimate()
        if cov.trace() > 1e-3:
            # 視覚信頼度低下 → 柔らかく
            self.imp.set_stiffness(reduce=True)

        goal = self.mpc.solve(self.current_state(), target=pose_cam)
        u = self.imp.compute_command(goal, self.latest_wrench)
        self.send_to_hw(u)
        self.log_metrics(pose_cam, u)


def main(args=None):
    rclpy.init(args=args)
    node = ImpedanceMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()