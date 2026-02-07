import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float32, Header
from geometry_msgs.msg import Point, PointStamped
from typing import Dict, Tuple
import math
import threading

class GaitConsensusNode(Node):
    def __init__(self) -> None:
        super().__init__('gait_consensus')

        # --- パラメータ ---
        self.declare_parameter('robot_id', 'robot0')
        self.declare_parameter('eps', 0.1)
        self.declare_parameter('safety_radius', 0.5)
        self.declare_parameter('com_frame', 'base_link')
        self.declare_parameter('neighbor_timeout', 0.5)

        self.robot_id: str = self.get_parameter('robot_id').value
        self.eps: float = self.get_parameter('eps').value
        self.safety_radius: float = self.get_parameter('safety_radius').value
        self.com_frame: str = self.get_parameter('com_frame').value
        self.neighbor_timeout: float = self.get_parameter('neighbor_timeout').value

        # --- 状態 ---
        self.phase: float = 0.0  # [0,1)
        self.coM: PointStamped = PointStamped()
        self.coM.header.frame_id = self.com_frame
        self.neighbors: Dict[str, Tuple[float, PointStamped, float]] = {}  # id -> (phase, com, stamp)

        self.lock = threading.Lock()

        # --- QoS ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # --- 通信 ---
        self.phase_pub = self.create_publisher(Float32, f'{self.robot_id}/gait_phase', qos)
        self.com_pub = self.create_publisher(PointStamped, f'{self.robot_id}/com_est', qos)

        self.create_subscription(
            Float32, 'gait_phase_in', self.on_phase, qos)
        self.create_subscription(
            PointStamped, 'com_est_in', self.on_com, qos)

        self.create_timer(0.05, self.step)  # 20 Hz
        self.create_timer(0.1, self.purge_neighbors)  # タイムアウト管理

    def on_phase(self, msg: Float32) -> None:
        with self.lock:
            self.neighbors[msg.header.frame_id] = (
                msg.data,
                self.neighbors.get(msg.header.frame_id, (0.0, PointStamped(), 0.0))[1],
                self.get_clock().now().nanoseconds * 1e-9
            )

    def on_com(self, msg: PointStamped) -> None:
        with self.lock:
            self.neighbors[msg.header.frame_id] = (
                self.neighbors.get(msg.header.frame_id, (0.0, PointStamped(), 0.0))[0],
                msg,
                self.get_clock().now().nanoseconds * 1e-9
            )

    def step(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        with self.lock:
            # 共有可能隣人のみ抽出
            valid = [(p, com) for rid, (p, com, t) in self.neighbors.items()
                     if now - t < self.neighbor_timeout]

            # 位相コンセンサス
            if valid:
                mean_diff = sum((p - self.phase) for p, _ in valid)
                self.phase += self.eps * mean_diff / len(valid)
                self.phase = self.phase % 1.0

            # 衝突回避
            for _, com in valid:
                if self.too_close(com):
                    self.get_logger().warn('Safety pause')
                    return

        # 公開
        self.phase_pub.publish(Float32(data=float(self.phase)))
        self.coM.header.stamp = self.get_clock().now().to_msg()
        self.com_pub.publish(self.coM)

    def too_close(self, other: PointStamped) -> bool:
        dx = self.coM.point.x - other.point.x
        dy = self.coM.point.y - other.point.y
        dz = self.coM.point.z - other.point.z
        return (dx*dx + dy*dy + dz*dz) < self.safety_radius**2

    def purge_neighbors(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        with self.lock:
            expired = [rid for rid, (_, _, t) in self.neighbors.items()
                       if now - t > self.neighbor_timeout]
            for rid in expired:
                del self.neighbors[rid]

def main(args=None):
    rclpy.init(args=args)
    node = GaitConsensusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()