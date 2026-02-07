#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Float32
from geometry_msgs.msg import PoseStamped
from example_interfaces.msg import Float64MultiArray  # 仮のアクション型
from builtin_interfaces.msg import Time
import time
import json

R_MAX = 10.0  # リスク閾値（適宜調整）

class EthicsMonitor(Node):
    def __init__(self):
        super().__init__('ethics_monitor')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.human_pose = None
        self.actuator_health = 1.0

        self.sub_action = self.create_subscription(
            Float64MultiArray, 'proposed_action', self.on_proposed_action, qos)
        self.sub_pose = self.create_subscription(
            PoseStamped, 'human_pose', self.on_human_pose, qos)
        self.sub_health = self.create_subscription(
            Float32, 'actuator_health', self.on_actuator_health, qos)

        self.pub_approved = self.create_publisher(
            Float64MultiArray, 'approved_action', qos)
        self.pub_audit = self.create_publisher(String, 'audit_log', 10)

    def on_human_pose(self, msg):
        self.human_pose = msg

    def on_actuator_health(self, msg):
        self.actuator_health = msg.data

    def compute_risk(self, action):
        if self.human_pose is None:
            return float('inf')  # 人間位置不明は最大リスク
        d = self._min_distance_to_human(action)
        sigma = self._pose_uncertainty_norm()
        eta = self.actuator_health
        return 1.0/(d+0.01) + sigma*0.5 + (1-eta)*2.0

    def _min_distance_to_human(self, action):
        # アクションと人間の距離簡易推定（Euclidean）
        return 1.0  # 実装依存で置換

    def _pose_uncertainty_norm(self):
        # 位置推定の不確実性ノルム（簡易固定値）
        return 0.1

    def safe_fallback(self):
        # 安全な停止指令（ゼロ速度）
        fallback = Float64MultiArray()
        fallback.data = [0.0]*6
        return fallback

    def on_proposed_action(self, msg):
        risk = self.compute_risk(msg.data)
        if risk > R_MAX:
            self.publish_audit(msg, risk, "rejected")
            approved = self.safe_fallback()
        else:
            self.publish_audit(msg, risk, "approved")
            approved = msg
        self.pub_approved.publish(approved)

    def publish_audit(self, msg, risk, status):
        log = {
            "timestamp": self.get_clock().now().to_msg().sec,
            "proposed": list(msg.data),
            "risk": risk,
            "status": status
        }
        audit_msg = String()
        audit_msg.data = json.dumps(log)
        self.pub_audit.publish(audit_msg)

def main(args=None):
    rclpy.init(args=args)
    node = EthicsMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()