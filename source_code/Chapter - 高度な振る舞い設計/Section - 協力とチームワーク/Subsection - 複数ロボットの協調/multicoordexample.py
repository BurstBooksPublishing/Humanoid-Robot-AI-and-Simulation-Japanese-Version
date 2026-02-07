import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
import argparse
import sys

# QoS設定：信頼性重視でネットワーク負荷を抑制
QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE
)

ALPHA = 0.1          # 合意ゲイン
PUBLISH_RATE = 10.0  # Hz
TIMEOUT_SEC = 1.0    # 隣接情報の有効期限[s]

class FormationConsensus(Node):
    def __init__(self, robot_id: int):
        super().__init__(f'formation_consensus_{robot_id}')
        self.robot_id = robot_id
        self.offset = 0.0
        self.neighbor_offsets = {}  # neighbor_id -> (offset, stamp)

        self.pub = self.create_publisher(Float32, '/desired_offset', QOS)
        self.create_subscription(
            PoseStamped, '/robot_state', self.state_cb, QOS)
        self.timer = self.create_timer(1.0 / PUBLISH_RATE, self.loop_cb)

    def state_cb(self, msg: PoseStamped):
        # frame_idに隣接ID、pose.xにオフセットを格納
        try:
            nid = int(msg.header.frame_id)
        except ValueError:
            self.get_logger().warn('frame_id is not integer')
            return
        self.neighbor_offsets[nid] = (msg.pose.position.x,
                                      self.get_clock().now())

    def loop_cb(self):
        now = self.get_clock().now()
        # タイムアウトした隣接を除去
        self.neighbor_offsets = {
            k: (v, t) for k, (v, t) in self.neighbor_offsets.items()
            if (now - t).nanoseconds * 1e-9 < TIMEOUT_SEC
        }

        if not self.neighbor_offsets:
            self.get_logger().warn('no valid neighbors')
            self.pub.publish(Float32(data=float(self.offset)))
            return

        # 合意更新
        err = sum(v - self.offset for v, _ in self.neighbor_offsets.values())
        self.offset += ALPHA * err / len(self.neighbor_offsets)

        self.pub.publish(Float32(data=float(self.offset)))

def main(argv=sys.argv):
    rclpy.init(args=argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-id', type=int, required=True)
    args, _ = parser.parse_known_args()

    node = FormationConsensus(args.robot_id)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()