#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from typing import List, Optional
import statistics
import time

# QoS設定：センサデータ用の信頼性・履歴ポリシー
QOS_PROFILE = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

class JitterMonitor(Node):
    def __init__(self) -> None:
        super().__init__('jitter_monitor')
        self._sub = self.create_subscription(
            JointState, '/joint_states', self._cb, QOS_PROFILE)
        self._prev_ns: Optional[int] = None
        self._deltas: List[float] = []
        self._window_size = 200  # 統計計算ウィンドウ
        self._logger = self.get_logger()

    def _cb(self, msg: JointState) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if self._prev_ns is not None:
            dt = (now_ns - self._prev_ns) * 1e-9
            self._deltas.append(dt)
            if len(self._deltas) >= self._window_size:
                mean = statistics.fmean(self._deltas)
                std = statistics.stdev(self._deltas)
                self._logger.info(f'avg={mean:.6f}s std={std:.6f}s')
                self._deltas.clear()
        self._prev_ns = now_ns

def main(args=None) -> None:
    rclpy.init(args=args)
    node = JitterMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()