import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Header
import time
import argparse

class HeartbeatNode(Node):
    def __init__(self, hz: int = 100):
        super().__init__('heartbeat')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.pub = self.create_publisher(Header, 'heartbeat', qos)
        self.sub = self.create_subscription(Header, 'heartbeat_echo', self.on_echo, qos)
        self.timer = self.create_timer(1.0 / hz, self.send)
        self.declare_parameter('frame_id', 'heartbeat')
        self.frame_id = self.get_parameter('frame_id').value
        self.get_logger().info(f'Heartbeat running at {hz} Hz')

    def send(self):
        msg = Header()
        msg.stamp = self.get_clock().now().to_msg()
        msg.frame_id = self.frame_id
        self.pub.publish(msg)

    def on_echo(self, msg: Header):
        now_ns = self.get_clock().now().nanoseconds
        sent_ns = msg.stamp.sec * 1_000_000_000 + msg.stamp.nanosec
        latency_ms = (now_ns - sent_ns) / 1e6
        self.get_logger().info(f'RTT: {latency_ms:.3f} ms')

def main(args=None):
    rclpy.init(args=args)
    node = HeartbeatNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()