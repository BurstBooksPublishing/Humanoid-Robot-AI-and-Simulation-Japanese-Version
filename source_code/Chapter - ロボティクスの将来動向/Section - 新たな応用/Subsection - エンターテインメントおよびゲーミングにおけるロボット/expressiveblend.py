import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from typing import Tuple, Optional

class GestureEngine(Node):
    def __init__(self) -> None:
        super().__init__('gesture_engine')

        # QoS設定（リアルタイム制御に適したプロファイル）
        rt_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # パラメータ読み込み
        self.declare_parameter('loop_hz', 200)
        self.declare_parameter('joint_names', [
            'shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3'
        ])
        self.loop_hz = self.get_parameter('loop_hz').value
        self.dt = 1.0 / self.loop_hz
        self.joint_names = self.get_parameter('joint_names').value

        # パブリッシャ／サブスクライバ
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', rt_qos
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, rt_qos
        )
        self.estop_pub = self.create_publisher(Bool, '/emergency_stop', rt_qos)

        # 内部状態
        self.latest_state: Optional[JointState] = None
        self.lock = threading.Lock()

        # 外部インターフェース（ダミー実装、実機ではROSトピック／サービスに置換）
        self.sensor_interface = SensorInterface()
        self.controller = Controller(self.joint_names)
        self.planner = Planner()

        # 制御ループタイマー
        self.timer = self.create_timer(self.dt, self.control_loop)

    def joint_state_cb(self, msg: JointState) -> None:
        with self.lock:
            self.latest_state = msg

    def control_loop(self) -> None:
        t0 = self.get_clock().now()

        # 最新の関節状態取得
        with self.lock:
            if self.latest_state is None:
                return
            state = np.array(self.latest_state.position)

        # センサ状態読み取り
        full_state = self.sensor_interface.read_state()
        audience = self.sensor_interface.estimate_audience()

        # 観客姿勢の遅延補償
        pred_aud = audience.pose + audience.vel * audience.measured_latency

        # 目標関節角度と表情ウェイト取得
        q_d, expr_w = self.planner.get_target(pred_aud)

        # マイクロ表情オーバーレイをブレンド
        micro = self.planner.micro_expression(expr_w)
        q_blend = q_d * (1 - expr_w) + (q_d + micro) * expr_w

        # トルク計算
        tau = self.controller.compute_torque(q_blend, full_state)

        # 指令送信
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = q_blend.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(self.dt * 1e9)
        traj.points.append(point)
        self.joint_cmd_pub.publish(traj)

        # 安全チェック
        if not self.sensor_interface.ok():
            self.estop_pub.publish(Bool(data=True))
            self.get_logger().error('Emergency stop triggered!')
            return

        # 周期維持
        elapsed = (self.get_clock().now() - t0).nanoseconds * 1e-9
        sleep_time = self.dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


class SensorInterface:
    def read_state(self):
        # 実機ではROSトピックから状態を取得
        return type('State', (), {
            'encoders': np.zeros(6),
            'imu': np.eye(3),
            'vision': np.zeros(3)
        })()

    def estimate_audience(self):
        # 実機ではVisionトピックから推定
        return type('Audience', (), {
            'pose': np.array([1.0, 0.0, 0.0]),
            'vel': np.array([0.1, 0.0, 0.0]),
            'measured_latency': 0.05
        })()

    def ok(self) -> bool:
        # センサ死活監視
        return True


class Controller:
    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.kp = np.array([100.0] * len(joint_names))
        self.kd = np.array([10.0] * len(joint_names))

    def compute_torque(self, q_d, state):
        # PD + フィードフォワード（簡易実装）
        q = state.encoders
        dq = np.zeros_like(q)  # 実機では速度取得
        return self.kp * (q_d - q) - self.kd * dq


class Planner:
    def get_target(self, pred_aud):
        # ジェスチャプリミティブ選択（ダミー）
        q_d = np.array([0.0, -0.5, 1.0, -1.5, 0.0, 0.0])
        expr_w = 0.3
        return q_d, expr_w

    def micro_expression(self, expr_w):
        # 微小オフセット生成
        return np.random.uniform(-0.05, 0.05, 6) * expr_w


def main(args=None):
    rclpy.init(args=args)
    node = GestureEngine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()