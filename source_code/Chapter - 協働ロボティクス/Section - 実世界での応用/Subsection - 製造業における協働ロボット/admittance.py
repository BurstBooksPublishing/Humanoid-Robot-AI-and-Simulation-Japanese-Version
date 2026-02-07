import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
import numpy as np
import threading
import time

class AdmittanceController(Node):
    def __init__(self):
        super().__init__('admittance_controller')

        # 仮想バネ・ダンパ・質量パラメータ
        self.M = 0.5
        self.B = 50.0
        self.K = 200.0
        self.dt = 0.01

        # 内部状態
        self.v = 0.0
        self.x = 0.0
        self.F_ext = 0.0
        self.pose_uncertainty = 0.0

        # 安全定数
        self.v_nominal = 0.4
        self.F_MAX = 20.0  # 許容最大外力

        # QoS: センサデータはベストエフォートで最新のみ
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 外力購読
        self.force_sub = self.create_subscription(
            Float64, 'wrist_force', self.force_cb, qos)

        # 姿勢不確かさ購読
        self.uncertainty_sub = self.create_subscription(
            Float64, 'pose_uncertainty', self.uncertainty_cb, qos)

        # 速度指令出版
        self.vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)

        # 周期タイマ
        self.timer = self.create_timer(self.dt, self.control_cycle)

        # スレッドセーフ用ロック
        self.lock = threading.Lock()

    def force_cb(self, msg: Float64):
        with self.lock:
            self.F_ext = msg.data

    def uncertainty_cb(self, msg: Float64):
        with self.lock:
            self.pose_uncertainty = msg.data

    def safety_check(self, force: float, pose_unc: float) -> float:
        # 速度上限を外力と不確かさで調整
        vmax = min(self.v_nominal,
                   max(0.05, self.F_MAX / (self.B + 10.0 * pose_unc)))
        return vmax

    def control_cycle(self):
        with self.lock:
            F = self.F_ext
            unc = self.pose_uncertainty
            vmax = self.safety_check(F, unc)

            # 離散アドミタンス更新（Euler積分）
            v_dot = (F - self.K * self.x - self.B * self.v) / self.M
            self.v += v_dot * self.dt
            self.v = np.clip(self.v, -vmax, vmax)
            self.x += self.v * self.dt

            # 速度指令メッセージ生成
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.twist.linear.x = self.v
            self.vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = AdmittanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()