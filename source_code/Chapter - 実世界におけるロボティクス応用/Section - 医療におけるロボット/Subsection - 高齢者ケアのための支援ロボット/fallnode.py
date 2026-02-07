#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
import threading

class FallDetector(Node):
    def __init__(self) -> None:
        super().__init__('fall_detector')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self._imu_sub = self.create_subscription(
            Imu, '/wearable/imu', self._imu_cb, qos)
        self._vision_sub = self.create_subscription(
            Bool, '/vision/fall_flag', self._vision_cb, qos)
        self._trigger_pub = self.create_publisher(
            Bool, '/assist/trigger', 10)

        self._imu_fall = False
        self._vision_fall = False
        self._lock = threading.Lock()

        self._timer = self.create_timer(0.1, self._timer_cb)  # 10 Hz

    def _imu_cb(self, msg: Imu) -> None:
        acc = (msg.linear_acceleration.x ** 2 +
               msg.linear_acceleration.y ** 2 +
               msg.linear_acceleration.z ** 2) ** 0.5
        with self._lock:
            self._imu_fall = acc > 25.0  # 閾値：25 m/s²

    def _vision_cb(self, msg: Bool) -> None:
        with self._lock:
            self._vision_fall = msg.data

    def _timer_cb(self) -> None:
        with self._lock:
            trigger = (self._imu_fall and self._vision_fall) or self._vision_fall
        self._trigger_pub.publish(Bool(data=trigger))

def main(args=None) -> None:
    rclpy.init(args=args)
    node = FallDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()