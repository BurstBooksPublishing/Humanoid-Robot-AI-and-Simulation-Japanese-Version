import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
import math
import time
import threading
from typing import Optional

class ImuMotorBridge(Node):
    def __init__(self) -> None:
        super().__init__('imu_motor_bridge')

        self.declare_parameter('imu_frame_id', 'imu_link')
        self.declare_parameter('publish_rate_hz', 200.0)
        self.declare_parameter('use_sim_time', False)

        self._imu_frame: str = self.get_parameter('imu_frame_id').value
        self._rate: float = self.get_parameter('publish_rate_hz').value
        self._period: float = 1.0 / self._rate

        self._cb_group_imu = ReentrantCallbackGroup()
        self._cb_group_cmd = MutuallyExclusiveCallbackGroup()

        imu_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            deadline=Duration(nanoseconds=10_000_000)
        )
        cmd_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE
        )

        self._imu_pub = self.create_publisher(
            Imu, '/imu/data_raw', imu_qos, callback_group=self._cb_group_imu
        )
        self._cmd_sub = self.create_subscription(
            String, '/motor/commands', self._cmd_callback, cmd_qos,
            callback_group=self._cb_group_cmd
        )

        self._timer = self.create_timer(
            self._period, self._publish_imu, callback_group=self._cb_group_imu
        )

        self._last_cmd: Optional[str] = None
        self._lock = threading.Lock()

        self.get_logger().info(f'IMU bridge ready. Frame={self._imu_frame}, Rate={self._rate} Hz')

    def _publish_imu(self) -> None:
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._imu_frame

        # ダミー値：実機ではドライバから取得
        msg.angular_velocity = Vector3(x=0.0, y=0.0, z=0.0)
        msg.linear_acceleration = Vector3(x=0.0, y=0.0, z=9.81)
        msg.angular_velocity_covariance[0] = -1.0
        msg.linear_acceleration_covariance[0] = -1.0

        self._imu_pub.publish(msg)

    def _cmd_callback(self, msg: String) -> None:
        with self._lock:
            self._last_cmd = msg.data
        # 高速パス：コマンドを即座にアクチュエータへ転送（実装省略）
        # 必要に応じてCAN/USB送信を別スレッドで実行

def main(args=None):
    rclpy.init(args=args)
    node = ImuMotorBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()