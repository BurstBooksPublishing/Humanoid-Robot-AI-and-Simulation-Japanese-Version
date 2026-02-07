#!/usr/bin/env python3
import json
from typing import Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped


class GrootBridge(Node):
    """Groot ブラックボード同期・スキル指令ブリッジ"""

    def __init__(self) -> None:
        super().__init__("groot_bridge")

        # QoS: 信頼性重視で transient_local（後からSubscribeしても最新を得る）
        bb_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ブラックボード更新購読
        self.create_subscription(
            Bool, "perception/object_visible", self._percept_cb, 10
        )

        # ブラックボード公開（Groot モニタ等が参照）
        self._bb_pub = self.create_publisher(String, "blackboard", bb_qos)

        # 把持指令
        self._grasp_pub = self.create_publisher(PoseStamped, "cmd/grasp_pose", 10)

        # スキル状態購読
        self.create_subscription(
            String, "skill/grasp/status", self._grasp_status_cb, 10
        )

        # 内部ブラックボード
        self._blackboard: Dict[str, bool] = {"object_visible": False}
        self._publish_blackboard()  # 初回送信

    # --- callbacks ----------------------------------------------------------
    def _percept_cb(self, msg: Bool) -> None:
        self._blackboard["object_visible"] = bool(msg.data)
        self._publish_blackboard()

    def _grasp_status_cb(self, msg: String) -> None:
        # 必要に応じて BT へ転送（本実装ではログ出力のみ）
        self.get_logger().debug(f"Grasp status: {msg.data}")

    # --- public API ---------------------------------------------------------
    def request_grasp(self, pose: PoseStamped) -> None:
        """外部スキルノードから呼ばれる把持指令"""
        self._grasp_pub.publish(pose)

    # --- internal -----------------------------------------------------------
    def _publish_blackboard(self) -> None:
        # JSON 形式で送信（Groot 側でパースしやすく）
        self._bb_pub.publish(String(data=json.dumps(self._blackboard)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GrootBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()