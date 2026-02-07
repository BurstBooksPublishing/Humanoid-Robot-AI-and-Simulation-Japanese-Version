import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Float32, Header
import numpy as np
from typing import Optional

class GraspComplianceNode(Node):
    def __init__(self) -> None:
        super().__init__('grasp_compliance')

        # QoS: センサデータ用に信頼性を確保
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self._sub = self.create_subscription(
            PoseStamped, 'grasp_pose', self.on_grasp, qos)

        self._force_pub = self.create_publisher(
            WrenchStamped, 'cmd_cartesian_force', qos)
        self._stiffness_pub = self.create_publisher(
            Float32, 'cmd_cartesian_stiffness', qos)

        # パラメータ宣言＆取得
        self.declare_parameter('force_max', 15.0)          # N
        self.declare_parameter('stiffness_min', 0.5)       # kN/m
        self.declare_parameter('stiffness_max', 1.0)       # kN/m
        self.declare_parameter('confidence_topic', '')

        self.F_MAX: float = self.get_parameter('force_max').value
        self.K_MIN: float = self.get_parameter('stiffness_min').value
        self.K_MAX: float = self.get_parameter('stiffness_max').value

        # confidence購読（オプション）
        conf_topic: str = self.get_parameter('confidence_topic').value
        self._latest_conf: float = 1.0
        if conf_topic:
            self.create_subscription(
                Float32, conf_topic, self.on_confidence, qos)

        self.get_logger().info("GraspComplianceNode ready.")

    def on_confidence(self, msg: Float32) -> None:
        # confidenceをキャッシュ
        self._latest_conf = max(0.0, min(1.0, msg.data))

    def on_grasp(self, msg: PoseStamped) -> None:
        # confidenceに応じて剛性をスケジューリング
        c = self._latest_conf
        K = self.K_MAX - (self.K_MAX - self.K_MIN) * (1.0 - c)
        K = float(np.clip(K, self.K_MIN, self.K_MAX))

        # 力指令はゼロ、制御器がF_MAXを守る
        wrench = WrenchStamped(
            header=Header(stamp=self.get_clock().now().to_msg(),
                          frame_id=msg.header.frame_id))
        # すべての成分をゼロに（既初期化済み）

        stiffness_msg = Float32(data=K)

        self._stiffness_pub.publish(stiffness_msg)
        self._force_pub.publish(wrench)

def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = GraspComplianceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()