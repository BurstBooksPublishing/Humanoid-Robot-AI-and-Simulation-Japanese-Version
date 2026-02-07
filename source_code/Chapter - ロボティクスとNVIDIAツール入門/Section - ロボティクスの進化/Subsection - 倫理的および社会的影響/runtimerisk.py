#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, Bool
from builtin_interfaces.msg import Time
import time
import logging
from typing import Optional

# QoS設定：センサデータ用BestEffort、ログ用Reliable
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)
LOG_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# パラメータ定義
THRESHOLD = 0.7
AUTONOMY_HIGH = 1.0
AUTONOMY_LOW = 0.2
LOOP_HZ = 50.0  # 50Hz制御ループ
H_PHYSICAL = 10.0  # 重大度重み


class RiskEstimator(Node):
    def __init__(self) -> None:
        super().__init__('risk_estimator')

        # パラメータ宣言と取得
        self.declare_parameter('threshold', THRESHOLD)
        self.declare_parameter('loop_hz', LOOP_HZ)
        self.threshold = self.get_parameter('threshold').value
        self.timer_period = 1.0 / self.get_parameter('loop_hz').value

        # サブスクライバ：センサ信頼度・近接距離
        self.conf_sub = self.create_subscription(
            Float32, '/perception/confidence', self.conf_callback, SENSOR_QOS)
        self.prox_sub = self.create_subscription(
            Float32, '/perception/proximity', self.prox_callback, SENSOR_QOS)

        # パブリッシャ：自律性レベル・リスク・安全挙動要求
        self.autonomy_pub = self.create_publisher(
            Float32, '/control/autonomy_level', LOG_QOS)
        self.risk_pub = self.create_publisher(
            Float32, '/risk/value', LOG_QOS)
        self.safe_req_pub = self.create_publisher(
            Bool, '/behavior/safe_request', LOG_QOS)

        # タイマー：メインループ
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # 内部状態
        self.confidence: float = 1.0
        self.proximity: float = 10.0
        self.autonomy: float = AUTONOMY_HIGH

        # ロガー設定
        self.logger = logging.getLogger('risk_estimator')
        self.logger.setLevel(logging.INFO)

    def conf_callback(self, msg: Float32) -> None:
        self.confidence = max(0.0, min(1.0, msg.data))

    def prox_callback(self, msg: Float32) -> None:
        self.proximity = max(0.01, msg.data)  # 0除算防止

    def expected_harm(self) -> float:
        # 物理的危害確率を計算
        p_physical = max(0.0, 1.0 - self.confidence) * (1.0 / self.proximity)
        return p_physical * H_PHYSICAL

    def timer_callback(self) -> None:
        risk = self.expected_harm()
        if risk > self.threshold:
            self.autonomy = AUTONOMY_LOW
            self.safe_req_pub.publish(Bool(data=True))
        else:
            self.autonomy = AUTONOMY_HIGH
            self.safe_req_pub.publish(Bool(data=False))

        # パブリッシュ
        self.autonomy_pub.publish(Float32(data=self.autonomy))
        self.risk_pub.publish(Float32(data=risk))

        # ログ（throttle 1Hz）
        self.get_logger().info(f'risk={risk:.3f} autonomy={self.autonomy}',
                               throttle_duration_sec=1.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RiskEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()