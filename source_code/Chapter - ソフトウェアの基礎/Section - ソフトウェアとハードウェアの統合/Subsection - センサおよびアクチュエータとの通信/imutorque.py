import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
import numpy as np
from typing import List
import threading
from rclpy.executors import MultiThreadedExecutor


class ImuTorqueController(Node):
    def __init__(self) -> None:
        super().__init__('imu_torque_controller')

        # QoS: センサデータ用ベストエフォート
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self._sub = self.create_subscription(
            Imu, 'imu/data', self.imu_cb, qos)

        self._pub = self.create_publisher(
            Float64MultiArray, 'joint_torques', 10)

        # パラメータ宣言＋取得
        self.declare_parameter('dt', 0.002)
        self.declare_parameter('alpha', 0.98)
        self.declare_parameter('kp', 30.0)
        self.declare_parameter('max_torque', 50.0)

        self.dt: float = self.get_parameter('dt').value
        self.alpha: float = self.get_parameter('alpha').value
        self.kp: float = self.get_parameter('kp').value
        self.max_torque: float = self.get_parameter('max_torque').value

        self.pitch: float = 0.0
        self.lock = threading.Lock()

    def imu_cb(self, msg: Imu) -> None:
        # 計算は別スレッドでブロックしない
        threading.Thread(target=self._compute_and_publish,
                         args=(msg,), daemon=True).start()

    def _compute_and_publish(self, msg: Imu) -> None:
        gz = msg.angular_velocity.y
        acc = msg.linear_acceleration
        pitch_acc = np.arctan2(-acc.x, np.hypot(acc.y, acc.z))

        with self.lock:
            self.pitch = self.alpha * (self.pitch + gz * self.dt) + \
                         (1.0 - self.alpha) * pitch_acc
            tau = -self.kp * self.pitch

        # 左右ヒップ対称
        torques: List[float] = [max(min(tau, self.max_torque), -self.max_torque),
                                max(min(tau, self.max_torque), -self.max_torque)]

        cmd = Float64MultiArray()
        cmd.data = torques
        self._pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ImuTorqueController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()