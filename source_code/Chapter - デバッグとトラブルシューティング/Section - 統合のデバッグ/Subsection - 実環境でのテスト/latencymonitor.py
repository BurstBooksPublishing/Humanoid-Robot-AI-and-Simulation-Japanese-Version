import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import csv
import time
import os
from datetime import datetime
from typing import Optional, Tuple

class LatencyMonitor(Node):
    def __init__(self) -> None:
        super().__init__('latency_monitor')

        # QoS設定：信頼性を高めて遅延を正確に測定
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub = self.create_subscription(
            JointState, '/joint_states', self.cb, qos)
        self.safe_pub = self.create_publisher(Bool, '/safe_stop', 10)

        self.declare_parameter('latency_threshold', 0.05)
        self.declare_parameter('log_dir', '/tmp/latency_logs')
        self.threshold: float = self.get_parameter(
            'latency_threshold').get_parameter_value().double_value

        # Welfordオンライン分散計算
        self._n: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0

        # ログファイル初期化
        log_dir: str = self.get_parameter(
            'log_dir').get_parameter_value().string_value
        os.makedirs(log_dir, exist_ok=True)
        timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path: str = os.path.join(log_dir, f'latency_{timestamp}.csv')
        self._csv_file = open(log_path, 'w', newline='')
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(['recv_time_sec', 'msg_time_sec', 'latency_sec'])

        self.get_logger().info(f'LatencyMonitor started, threshold={self.threshold}s')

    def cb(self, msg: JointState) -> None:
        now = self.get_clock().now()
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        latency_sec = (now - msg_time).nanoseconds * 1e-9

        # オンライン統計更新
        self._n += 1
        delta = latency_sec - self._mean
        self._mean += delta / self._n
        self._M2 += delta * (latency_sec - self._mean)

        self._writer.writerow([now.nanoseconds * 1e-9,
                               msg_time.nanoseconds * 1e-9,
                               latency_sec])

        if latency_sec > self.threshold:
            self.get_logger().warn(
                f'Latency exceeded: {latency_sec:.4f}s > {self.threshold}s')
            self.safe_pub.publish(Bool(data=True))

    def get_stats(self) -> Tuple[float, float]:
        if self._n < 2:
            return self._mean, 0.0
        var = self._M2 / (self._n - 1)
        return self._mean, var ** 0.5

    def destroy_node(self) -> None:
        self._csv_file.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LatencyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        mean, stddev = node.get_stats()
        node.get_logger().info(
            f'Shutting down: samples={node._n}, mean={mean:.4f}s, std={stddev:.4f}s')
        node.destroy_node()
        rclpy.shutdown()