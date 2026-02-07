import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, TwistStamped
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import math

class CapturePointNode(Node):
    def __init__(self) -> None:
        super().__init__('capture_point_node')

        # パラメータ宣言と取得
        self.declare_parameter(
            'com_height',
            0.9,
            ParameterDescriptor(
                description='公称重心高さ [m]',
                type=ParameterType.PARAMETER_DOUBLE,
                read_only=True
            )
        )
        self.zc: float = self.get_parameter('com_height').value
        self.g: float = 9.81

        # パブリッシャ／サブスクライバ
        self._pub = self.create_publisher(Point, '~/capture_point', 10)
        self._sub = self.create_subscription(
            TwistStamped, '~/com_state', self._cb_state, 10
        )

        self.get_logger().info(f'CapturePointNode started (zc={self.zc})')

    def _cb_state(self, msg: TwistStamped) -> None:
        # 重心速度と位置を TwistStamped から取得
        v = msg.twist.linear.x
        x = msg.twist.linear.z  # zフィールドをx座標に流用

        omega = math.sqrt(self.g / self.zc)
        x_cp = x + v / omega

        pt = Point()
        pt.x = x_cp
        pt.y = 0.0
        pt.z = 0.0
        self._pub.publish(pt)

def main(args=None):
    rclpy.init(args=args)
    node = CapturePointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()