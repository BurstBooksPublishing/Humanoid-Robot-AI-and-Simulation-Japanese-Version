#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray
import math
import time
from typing import List, Optional
from threading import Lock

class MaintenanceNode(Node):
    def __init__(self) -> None:
        super().__init__('maintenance_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._sub = self.create_subscription(
            Float32MultiArray, '/telemetry/motors', self._telemetry_cb, qos)

        self._failures: List[float] = []
        self._lock = Lock()
        self._last_fail_t: Optional[float] = None
        self._current_t: float = 0.0

        # 閾値はパラメータで変更可能
        self.declare_parameter('current_threshold', 50.0)
        self.declare_parameter('p_max', 0.01)
        self.declare_parameter('min_failures', 2)

        self._timer = self.create_timer(1.0, self._publish_inspection_interval)

    def _telemetry_cb(self, msg: Float32MultiArray) -> None:
        threshold = self.get_parameter('current_threshold').value
        if max(msg.data) > threshold:
            with self._lock:
                now = time.monotonic() / 3600.0  # 時間単位
                if self._last_fail_t is None or (now - self._last_fail_t) > 1e-3:
                    self._failures.append(now)
                    self._last_fail_t = now

    def _publish_inspection_interval(self) -> None:
        with self._lock:
            if len(self._failures) < self.get_parameter('min_failures').value:
                return
            intervals = [self._failures[i+1] - self._failures[i]
                         for i in range(len(self._failures)-1)]
            mtbf = sum(intervals) / len(intervals)
            lam = 1.0 / mtbf
            p_max = self.get_parameter('p_max').value
            t_i = -math.log(1.0 - p_max) / lam
            self.get_logger().info(f'Inspection interval: {t_i:.2f} h')

def main(args=None):
    rclpy.init(args=args)
    node = MaintenanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()