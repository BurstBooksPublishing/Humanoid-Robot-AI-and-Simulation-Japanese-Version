import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import Imu
import spidev
import struct
import time
from typing import List, Tuple

class ImuPublisher(Node):
    def __init__(self) -> None:
        super().__init__('imu_publisher')

        # パラメータ宣言
        self.declare_parameter('frame_id', 'imu_link',
                               ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('bus', 0,
                               ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('device', 0,
                               ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('max_speed_hz', 1_000_000,
                               ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('timer_period_sec', 0.005,
                               ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('bias_gyro', [0.0, 0.0, 0.0],
                               ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))

        # パラメータ取得
        frame_id = self.get_parameter('frame_id').value
        bus = self.get_parameter('bus').value
        device = self.get_parameter('device').value
        max_speed_hz = self.get_parameter('max_speed_hz').value
        timer_period = self.get_parameter('timer_period_sec').value
        self.bias_gyro: List[float] = self.get_parameter('bias_gyro').value

        # Publisher
        self.pub = self.create_publisher(Imu, 'imu/data_raw', 10)

        # SPI初期化
        try:
            self.spi = spidev.SpiDev()
            self.spi.open(bus, device)
            self.spi.max_speed_hz = max_speed_hz
            self.spi.mode = 0b11
            self.get_logger().info(f'SPI初期化完了 bus={bus} device={device}')
        except Exception as e:
            self.get_logger().error(f'SPI初期化失敗: {e}')
            raise

        # タイマー
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # メッセージ雛形
        self.imu_msg = Imu()
        self.imu_msg.header.frame_id = frame_id

    def read_imu(self) -> Tuple[List[float], List[float]]:
        # レジスタアドレスとダミー読出し（センサ依存）
        try:
            raw = self.spi.xfer2([0x3B, 0x00] + [0x00] * 12)
            data = bytes(raw[2:])
            ax, ay, az, gx, gy, gz = struct.unpack('>6h', data)
            scale_acc = 1e-3
            scale_gyro = 1e-3
            return ([a * scale_acc for a in (ax, ay, az)],
                    [g * scale_gyro for g in (gx, gy, gz)])
        except Exception as e:
            self.get_logger().warn(f'SPI通信エラー: {e}')
            return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def timer_callback(self) -> None:
        acc, gyro = self.read_imu()
        gyro = [g - b for g, b in zip(gyro, self.bias_gyro)]

        self.imu_msg.header.stamp = self.get_clock().now().to_msg()
        self.imu_msg.linear_acceleration.x = acc[0]
        self.imu_msg.linear_acceleration.y = acc[1]
        self.imu_msg.linear_acceleration.z = acc[2]
        self.imu_msg.angular_velocity.x = gyro[0]
        self.imu_msg.angular_velocity.y = gyro[1]
        self.imu_msg.angular_velocity.z = gyro[2]
        self.pub.publish(self.imu_msg)

    def destroy_node(self) -> None:
        self.spi.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImuPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()