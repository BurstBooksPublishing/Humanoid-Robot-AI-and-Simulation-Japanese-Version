import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Vector3Stamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
from typing import Optional, Tuple
import threading
from collections import deque

class StateEstimate:
    """スレッドセーフな状態推定値のコンテナ"""
    __slots__ = ['timestamp', 'q', 'qd', 'qdd', 'rpy', 'omega']
    def __init__(self):
        self.timestamp: rclpy.time.Time = rclpy.time.Time()
        self.q: np.ndarray = np.zeros(12)      # 関節角度
        self.qd: np.ndarray = np.zeros(12)     # 関節速度
        self.qdd: np.ndarray = np.zeros(12)    # 関節加速度（差分近似）
        self.rpy: np.ndarray = np.zeros(3)     # ロール・ピッチ・ヨー
        self.omega: np.ndarray = np.zeros(3)   # 角速度

class EstimatorNode(Node):
    def __init__(self):
        super().__init__('estimator_node')

        # パラメータ宣言
        self.declare_parameter('queue_size', 30)
        self.declare_parameter('slop', 0.005)
        self.declare_parameter('alpha', 0.1)  # 補完係数
        self.add_on_set_parameters_callback(self.param_cb)

        # QoS: リアルタイム制御に適した設定
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscriber
        imu_sub = Subscriber(self, Imu, '/imu/data', qos_profile=qos)
        joint_sub = Subscriber(self, JointState, '/joint_states', qos_profile=qos)

        # TimeSynchronizer
        self.ats = ApproximateTimeSynchronizer(
            [imu_sub, joint_sub],
            queue_size=self.get_parameter('queue_size').value,
            slop=self.get_parameter('slop').value
        )
        self.ats.registerCallback(self.synced_callback)

        # Publisher
        self.state_pub = self.create_publisher(Vector3Stamped, '~/state/roll_pitch', qos)

        # リングバッファ（最新推定値をロックフリーで共有）
        self.state_buffer: deque = deque(maxlen=1)
        self.lock = threading.Lock()

        # 前回値保持（差分計算用）
        self.prev_joint: Optional[Tuple[np.ndarray, rclpy.time.Time]] = None

        self.get_logger().info("EstimatorNode initialized")

    def param_cb(self, params: list) -> SetParametersResult:
        for p in params:
            if p.name == 'slop':
                self.ats.set_slop(p.value)
        return SetParametersResult(successful=True)

    def synced_callback(self, imu_msg: Imu, joint_msg: JointState):
        state = StateEstimate()
        state.timestamp = imu_msg.header.stamp

        # IMUからロール・ピッチ・角速度を抽出
        orient = imu_msg.orientation
        # クォータンション → オイラー角（ZYX）
        sinr_cosp = 2.0 * (orient.w * orient.x + orient.y * orient.z)
        cosr_cosp = 1.0 - 2.0 * (orient.x * orient.x + orient.y * orient.y)
        state.rpy[0] = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (orient.w * orient.y - orient.z * orient.x)
        state.rpy[1] = np.arcsin(np.clip(sinp, -1.0, 1.0))

        state.omega = np.array([imu_msg.angular_velocity.x,
                                imu_msg.angular_velocity.y,
                                imu_msg.angular_velocity.z])

        # 関節状態をNumPy配列に変換
        q = np.array(joint_msg.position, dtype=np.float64)
        qd = np.array(joint_msg.velocity, dtype=np.float64)

        # 差分から加速度を近似（時刻差を考慮）
        if self.prev_joint is not None:
            prev_q, prev_stamp = self.prev_joint
            dt = (state.timestamp - prev_stamp).nanoseconds * 1e-9
            if dt > 1e-4:
                state.qdd = (qd - prev_q) / dt
        self.prev_joint = (qd, state.timestamp)

        state.q = q
        state.qd = qd

        # バッファ更新（ロック保護）
        with self.lock:
            self.state_buffer.append(state)

        # デバッグ出力（軽量化のため最小限）
        msg = Vector3Stamped()
        msg.header.stamp = state.timestamp
        msg.vector.x = state.rpy[0]
        msg.vector.y = state.rpy[1]
        msg.vector.z = state.omega[2]
        self.state_pub.publish(msg)

    def get_latest_state(self) -> Optional[StateEstimate]:
        """制御ループから呼ばれる：最新推定値を取得"""
        with self.lock:
            return self.state_buffer[-1] if self.state_buffer else None

def main(args=None):
    rclpy.init(args=args)
    node = EstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()