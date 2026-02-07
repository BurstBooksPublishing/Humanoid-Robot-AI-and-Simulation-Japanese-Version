import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch
import numpy as np
from typing import Optional, List

class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__('policy_node')

        # QoS設定：リアルタイム性と信頼性を両立
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.policy = torch.jit.load('policy.pt', map_location='cpu')
        self.policy.eval()  # 推論モード固定

        self.joint_names: List[str] = []  # 初期化後に設定
        self.create_subscription(
            JointState, '/joint_states', self.js_cb, qos)

        self.cmd_pub = self.create_publisher(
            JointTrajectory, '/joint_commands', qos)

        # 安全制限（Nm）
        self.torque_max = 50.0
        self.torque_limit = np.array([self.torque_max], dtype=np.float32)

        self.last_state: Optional[JointState] = None

    def js_cb(self, msg: JointState) -> None:
        self.last_state = msg
        if not self.joint_names:
            self.joint_names = msg.name  # 初回のみ記憶

        action = self.infer_action(msg)
        if self.check_safety(action):
            traj = self.mk_traj(action)
            self.cmd_pub.publish(traj)
        else:
            self.get_logger().warn('Safety limit exceeded, fallback to safe.')
            safe_traj = self.safe_controller(msg)
            self.cmd_pub.publish(safe_traj)

    def infer_action(self, js: JointState) -> np.ndarray:
        obs = torch.tensor(js.position + js.velocity, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a = self.policy(obs)
        return a.squeeze(0).cpu().numpy()

    def check_safety(self, action: np.ndarray) -> bool:
        return np.all(np.abs(action) <= self.torque_limit)

    def mk_traj(self, action: np.ndarray) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.effort = action.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(20e6)  # 20 ms
        traj.points.append(point)
        return traj

    def safe_controller(self, js: JointState) -> JointTrajectory:
        # 零トルク指令で安全停止
        traj = JointTrajectory()
        traj.joint_names = js.name
        point = JointTrajectoryPoint()
        point.effort = [0.0] * len(js.name)
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(20e6)
        traj.points.append(point)
        return traj

def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()