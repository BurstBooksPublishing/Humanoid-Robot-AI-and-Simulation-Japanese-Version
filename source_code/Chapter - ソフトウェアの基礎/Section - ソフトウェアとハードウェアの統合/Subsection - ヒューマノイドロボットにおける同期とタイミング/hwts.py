import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Imu
import time
import threading
from typing import Optional

class LatencyMonitor(Node):
    def __init__(self) -> None:
        super().__init__('latency_monitor')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._sub = self.create_subscription(
            Imu, 'imu/data_raw', self._imu_cb, qos)

        self._declare_parameters()
        self._threshold: float = self.get_parameter('latency_threshold_ms').value * 1e-3
        self._window_size: int = self.get_parameter('window_size').value
        self._log_period: float = self.get_parameter('log_period_sec').value

        self._lock = threading.Lock()
        self._latencies: list[float] = []
        self._last_log_time = time.monotonic()

        self._logger.info('LatencyMonitor started.')

    def _declare_parameters(self) -> None:
        self.declare_parameter('latency_threshold_ms', 5.0)
        self.declare_parameter('window_size', 100)
        self.declare_parameter('log_period_sec', 1.0)

    def _imu_cb(self, msg: Imu) -> None:
        now = time.monotonic()
        hw_sec = float(msg.header.sec) + float(msg.header.nanosec) * 1e-9
        latency = now - hw_sec

        with self._lock:
            self._latencies.append(latency)
            if len(self._latencies) > self._window_size:
                self._latencies.pop(0)

        if latency > self._threshold:
            self.get_logger().warn(
                f'High latency detected: {latency*1e3:.3f} ms')

        if now - self._last_log_time > self._log_period:
            self._last_log_time = now
            with self._lock:
                if self._latencies:
                    avg = sum(self._latencies) / len(self._latencies)
                    self.get_logger().info(
                        f'Average latency (last {len(self._latencies)}): {avg*1e3:.3f} ms')

def main(args=None) -> None:
    rclpy.init(args=args)
    node = LatencyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()