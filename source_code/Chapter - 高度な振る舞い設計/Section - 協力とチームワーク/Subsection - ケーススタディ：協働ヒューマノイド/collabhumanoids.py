import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import math

class Auctioneer(Node):
    def __init__(self):
        super().__init__('auctioneer')

        # パラメータ宣言
        self.declare_parameter('bid_rate', 10.0,
                               ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                   description='Bid publish frequency [Hz]'))
        self.declare_parameter('battery_topic', '/battery/percentage',
                               ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('reachability_radius', 2.0,
                               ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))

        # QoS設定
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST,
                         depth=10)

        # 購読・配信
        self.create_subscription(PoseStamped, '/payload/pose',
                                 self.pose_cb, qos)
        self.create_subscription(Float32, self.get_parameter('battery_topic').value,
                                 self.battery_cb, qos)
        self.bid_pub = self.create_publisher(Float32, '/allocator/bid', qos)

        # タイマー
        period = 1.0 / self.get_parameter('bid_rate').value
        self.timer = self.create_timer(period, self.publish_bid)

        # 状態変数
        self.latest_pose = None
        self.battery_pct = 1.0
        self.local_quality = 0.0

    def pose_cb(self, msg: PoseStamped):
        self.latest_pose = msg
        self.local_quality = self.estimate_quality(msg)

    def battery_cb(self, msg: Float32):
        self.battery_pct = max(0.0, min(1.0, msg.data))

    def estimate_quality(self, pose: PoseStamped) -> float:
        # 到達可能性とバッテリ残量を掛け合わせた品質指標
        reach = self._reachability(pose)
        return reach * self.battery_pct

    def _reachability(self, pose: PoseStamped) -> float:
        # 原点からの距離に応じて線形減衰
        r = self.get_parameter('reachability_radius').value
        d = math.hypot(pose.pose.position.x, pose.pose.position.y)
        return max(0.0, 1.0 - d / r) if r > 0.0 else 0.0

    def publish_bid(self):
        if self.latest_pose is None:
            return  # 初期データ未受信
        bid = Float32()
        bid.data = float(self.local_quality)
        self.bid_pub.publish(bid)

def main(args=None):
    rclpy.init(args=args)
    node = Auctioneer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()