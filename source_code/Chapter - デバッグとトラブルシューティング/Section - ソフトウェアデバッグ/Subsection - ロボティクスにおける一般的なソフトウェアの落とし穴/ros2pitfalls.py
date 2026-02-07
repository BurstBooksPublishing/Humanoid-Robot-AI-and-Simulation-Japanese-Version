import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import TwistStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
from typing import Optional, Tuple


class SensorFusionNode(Node):
    def __init__(self) -> None:
        super().__init__('sensor_fusion')

        # QoS: 信頼性を保ちつつリアルタイム性を重視
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # サブスクライバ
        self.imu_sub = Subscriber(self, Imu, '/imu', qos_profile=qos)
        self.joint_sub = Subscriber(self, JointState, '/joint_states', qos_profile=qos)

        # 同期器: スラック 10 ms, キューサイズ 50
        self.sync = ApproximateTimeSynchronizer(
            [self.imu_sub, self.joint_sub],
            queue_size=50,
            slop=0.01
        )
        self.sync.registerCallback(self.fused_callback)

        # パブリッシャ: 融合後の速度推定値を配信
        self.fused_pub = self.create_publisher(TwistStamped, '~/fused_twist', qos)

        # パラメータ
        self.declare_parameter('max_staleness_s', 0.05)
        self.max_staleness = Duration(seconds=self.get_parameter('max_staleness_s').value)

        # キャリブレーション用バッファ
        self.gyro_bias: Optional[np.ndarray] = None
        self.calib_count = 0
        self.calib_samples = 200

        self.get_logger().info('SensorFusionNode initialized')

    def fused_callback(self, imu: Imu, joints: JointState) -> None:
        now = self.get_clock().now()

        # 時刻チェック: 古いメッセージは破棄
        imu_time = Time.from_msg(imu.header.stamp)
        if (now - imu_time) > self.max_staleness:
            return

        # キャリブレーション
        if self.gyro_bias is None:
            self._calibrate_gyro(imu)
            return

        # バイアス除去後の角速度
        omega = np.array([
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z
        ]) - self.gyro_bias

        # ホイール半径・間隔のパラメータ（仮）
        wheel_radius = 0.08  # [m]
        wheel_base = 0.4     # [m]

        # ジョイント名から左右ホイール速度を抽出
        left_vel = right_vel = 0.0
        for name, vel in zip(joints.name, joints.velocity):
            if 'left' in name:
                left_vel = vel
            elif 'right' in name:
                right_vel = vel

        # 差動二輪モデルで速度推定
        v_left = left_vel * wheel_radius
        v_right = right_vel * wheel_radius
        v = (v_left + v_right) * 0.5
        w = (v_right - v_left) / wheel_base

        # 融合: 角速度はIMU優先, 並進速度はエンコーダ優先
        fused_w = omega[2] if abs(omega[2]) > 0.01 else w

        # パブリッシュ
        twist = TwistStamped()
        twist.header.stamp = imu.header.stamp
        twist.header.frame_id = 'base_link'
        twist.twist.linear.x = float(v)
        twist.twist.angular.z = float(fused_w)
        self.fused_pub.publish(twist)

    def _calibrate_gyro(self, imu: Imu) -> None:
        if self.calib_count == 0:
            self.get_logger().info('Calibrating gyro bias...')
            self.gyro_buffer: list = []

        self.gyro_buffer.append([
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z
        ])
        self.calib_count += 1

        if self.calib_count >= self.calib_samples:
            self.gyro_bias = np.mean(self.gyro_buffer, axis=0)
            self.get_logger().info(f'Gyro bias calibrated: {self.gyro_bias}')
            self.gyro_buffer.clear()


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()