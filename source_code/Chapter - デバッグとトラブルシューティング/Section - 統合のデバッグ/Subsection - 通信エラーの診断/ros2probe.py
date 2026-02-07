import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
import time
import statistics
from typing import List, Optional
import threading

class CommProbe(Node):
    def __init__(self,
                 topic: str = '/joint_states',
                 expected_rate: float = 100.0,
                 window_size: int = 1000) -> None:
        super().__init__('comm_probe')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._sub = self.create_subscription(
            JointState, topic, self._cb, qos)

        self._expected_dt = 1.0 / expected_rate
        self._window_size = window_size

        self._arrivals: List[float] = []
        self._latencies: List[float] = []
        self._last_recv: Optional[float] = None
        self._miss_count = 0
        self._recv_count = 0
        self._lock = threading.Lock()

        self._timer = self.create_timer(1.0, self._print_stats)

    def _cb(self, msg: JointState) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        pub_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        latency = now - pub_stamp
        with self._lock:
            self._latencies.append(latency)
            if self._last_recv is not None:
                dt = now - self._last_recv
                self._arrivals.append(dt)
                if dt > 1.5 * self._expected_dt:
                    self._miss_count += int(round(dt / self._expected_dt)) - 1
            self._last_recv = now
            self._recv_count += 1
            # リングバッファでメモリ使用量を制限
            if len(self._latencies) > self._window_size:
                self._latencies.pop(0)
            if len(self._arrivals) > self._window_size:
                self._arrivals.pop(0)

    def _print_stats(self) -> None:
        with self._lock:
            if not self._latencies or not self._arrivals:
                return
            self.get_logger().info(
                f"recv={self._recv_count} miss~={self._miss_count} "
                f"latency_mean={statistics.mean(self._latencies):.3f}s "
                f"jitter={statistics.pstdev(self._arrivals):.3f}s"
            )

def main(args=None) -> None:
    rclpy.init(args=args)
    probe = CommProbe()
    try:
        rclpy.spin(probe)
    except KeyboardInterrupt:
        pass
    finally:
        probe.destroy_node()
        rclpy.shutdown()