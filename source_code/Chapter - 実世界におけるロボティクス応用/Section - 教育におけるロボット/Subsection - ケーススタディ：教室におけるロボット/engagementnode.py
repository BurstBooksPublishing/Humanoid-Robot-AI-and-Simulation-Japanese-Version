import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32
from typing import Deque
from collections import deque
import yaml
import os
from ament_index_python.packages import get_package_share_directory

class EngagementNode(Node):
    def __init__(self) -> None:
        super().__init__('engagement_node')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._pub = self.create_publisher(Float32, 'engagement_score', qos)
        self.declare_parameter('timer_period', 0.5)
        self.declare_parameter('window_size', 10)
        self.declare_parameter('weights', [0.6, 0.3, 0.1])
        timer_period = self.get_parameter('timer_period').value
        self._window_size = self.get_parameter('window_size').value
        weights = self.get_parameter('weights').value
        self._w_vis, self._w_aud, self._w_prox = weights
        self._window: Deque[float] = deque(maxlen=self._window_size)
        self.create_timer(timer_period, self._timer_cb)
        self.get_logger().info("EngagementNode initialized")

    def _timer_cb(self) -> None:
        try:
            visual = self._read_visual_attention()
            audio = self._read_audio_activity()
            prox = self._read_proximity()
        except Exception as e:
            self.get_logger().warning(f"Sensor read failed: {e}")
            return
        # 重み付き融合 & クリップ
        score = max(0.0, min(1.0, self._w_vis * visual +
                                        self._w_aud * audio +
                                        self._w_prox * (1.0 - prox)))
        self._window.append(score)
        avg = sum(self._window) / len(self._window) if self._window else 0.0
        msg = Float32(data=float(avg))
        self._pub.publish(msg)

    def _read_visual_attention(self) -> float:
        # TODO: 実際の視線推定モジュールと接続
        return 0.8

    def _read_audio_activity(self) -> float:
        # TODO: 実際の音声検出モジュールと接続
        return 0.2

    def _read_proximity(self) -> float:
        # TODO: 実際の距離センサと接続
        return 0.1

def main(args=None):
    rclpy.init(args=args)
    node = EngagementNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()