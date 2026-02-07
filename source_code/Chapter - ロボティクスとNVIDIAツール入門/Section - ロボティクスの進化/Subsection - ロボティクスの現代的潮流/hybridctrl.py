import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import torch
from typing import Optional, Tuple

class ResidualPDController(Node):
    def __init__(self) -> None:
        super().__init__('residual_pd_controller')
        
        # QoS設定（リアルタイム制御向け）
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 関節状態購読
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, qos
        )
        
        # トルク指令出版
        self.torque_pub = self.create_publisher(
            Float64MultiArray, '/joint_torque_command', qos
        )
        
        # パラメータ宣言
        self.declare_parameter('kp', 200.0)
        self.declare_parameter('kd', 10.0)
        self.declare_parameter('alpha', 0.6)
        self.declare_parameter('n_joints', 7)
        
        # 内部状態
        self.q: Optional[np.ndarray] = None
        self.qdot: Optional[np.ndarray] = None
        self.qd: Optional[np.ndarray] = None
        
        # 学習済みポリシー読み込み（TorchScript）
        policy_path = self.declare_parameter('policy_path', '').value
        self.policy = torch.jit.load(policy_path) if policy_path else None
        
        # タイマー（1 kHz）
        self.timer = self.create_timer(0.001, self.control_loop)
        
    def joint_callback(self, msg: JointState) -> None:
        # 関節角度・速度をNumPy配列に変換
        self.q = np.array(msg.position, dtype=np.float64)
        self.qdot = np.array(msg.velocity, dtype=np.float64)
        
    def get_desired_state(self) -> Tuple[np.ndarray, np.ndarray]:
        # 目標関節角度（外部トピックまたは内部生成）
        if self.qd is None:
            n = self.get_parameter('n_joints').value
            self.qd = np.zeros(n, dtype=np.float64)
        return self.qd, np.zeros_like(self.qd)
        
    def control_loop(self) -> None:
        if self.q is None or self.qdot is None:
            return
            
        # 動力学計算（C++バックエンド）
        g = self.compute_gravity(self.q)
        M = self.compute_mass_matrix(self.q)
        
        # PDフィードバック
        qd, qd_dot = self.get_desired_state()
        kp = self.get_parameter('kp').value
        kd = self.get_parameter('kd').value
        tau_pd = kp * (qd - self.q) + kd * (qd_dot - self.qdot)
        
        # 学習残差推定
        alpha = self.get_parameter('alpha').value
        tau_learn = self.infer_residual(self.q, self.qdot) if self.policy else 0.0
        
        # 最終トルク指令
        tau_cmd = tau_pd + g + alpha * tau_learn
        
        # 指令出版
        msg = Float64MultiArray()
        msg.data = tau_cmd.tolist()
        self.torque_pub.publish(msg)
        
    def compute_gravity(self, q: np.ndarray) -> np.ndarray:
        # 実機ではC++ライブラリ呼び出しに置換
        return np.zeros_like(q)
        
    def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        # 実機ではC++ライブラリ呼び出しに置換
        n = len(q)
        return np.eye(n, dtype=np.float64)
        
    def infer_residual(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        # センサ観測を含む状態ベクトル構築
        obs = self.get_sensor_obs()
        state = torch.tensor(np.concatenate([q, qdot, obs]), dtype=torch.float32)
        with torch.no_grad():
            return self.policy(state).numpy()

    def get_sensor_obs(self) -> np.ndarray:
        # 追加センサ（力覚・視覚等）読み取り
        return np.array([], dtype=np.float64)

def main(args=None):
    rclpy.init(args=args)
    node = ResidualPDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()