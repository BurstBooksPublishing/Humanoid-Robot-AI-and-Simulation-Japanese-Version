import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import torch
import yaml
from pathlib import Path

class HumanoidEnv(gym.Env):
    """Production-ready humanoid environment with ROS 2 integration."""
    
    def __init__(self, sim, config_path: str = None):
        super().__init__()
        
        # 設定ファイル読み込み
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "humanoid_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sim = sim
        self.node = Node('humanoid_env')
        
        # ROS 2 publishers/subscribers
        self.obs_pub = self.node.create_publisher(Float32MultiArray, '/humanoid/observation', 10)
        self.reward_pub = self.node.create_publisher(Float32MultiArray, '/humanoid/reward', 10)
        self.done_pub = self.node.create_publisher(Bool, '/humanoid/done', 10)
        
        # Action/Observation spaces
        self.action_space = spaces.Box(
            low=np.array(self.config['action_limits']['low']),
            high=np.array(self.config['action_limits']['high']),
            dtype=np.float32
        )
        
        obs_dim = self.config['observation']['dim']
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # PD制御ゲイン
        self.Kp = np.array(self.config['control']['kp'])
        self.Kd = np.array(self.config['control']['kd'])
        self.torque_limit = self.config['control']['torque_limit']
        
        # 報酬重み
        self.reward_weights = self.config['reward']['weights']
        
        # 内部状態
        self.step_count = 0
        self.max_steps = self.config['episode']['max_steps']
        self.prev_action = np.zeros_like(self.Kp)
        
    def randomize_dynamics(self) -> None:
        """動的パラメータのランダマイゼーション（ドメインランダマイゼーション）"""
        # 質量のランダマイゼーション
        mass_scale = 1.0 + self.config['randomization']['mass_std'] * np.random.randn()
        self.sim.set_mass_scale(np.clip(mass_scale, 0.8, 1.2))
        
        # 摩擦係数のランダマイゼーション
        friction = self.config['randomization']['base_friction'] + \
                  self.config['randomization']['friction_range'] * np.random.rand()
        self.sim.set_friction(np.clip(friction, 0.5, 1.0))
        
        # 重心位置のランダマイゼーション
        com_offset = np.random.uniform(
            -self.config['randomization']['com_range'],
            self.config['randomization']['com_range'],
            size=3
        )
        self.sim.set_com_offset(com_offset)
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """環境のリセット"""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.prev_action = np.zeros_like(self.Kp)
        
        # シミュレーションリセット
        self.sim.reset()
        
        # ランダマイゼーション適用
        if self.config['randomization']['enabled']:
            self.randomize_dynamics()
            
        # 初期状態取得
        obs = self._compute_observation()
        
        # ROS 2メッセージ配信
        self._publish_observation(obs)
        
        return obs, {}
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """1ステップ実行"""
        # アクションのクリッピングと平滑化
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = 0.8 * action + 0.2 * self.prev_action  # 低周波フィルタ
        
        # 現在状態取得
        q, qdot = self.sim.get_proprioception()
        
        # PD制御によるトルク計算
        tau = self.Kp * (action - q) - self.Kd * qdot
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        
        # トルク適用とシミュレーションステップ
        self.sim.apply_torques(tau)
        self.sim.step()
        
        # 観測と報酬計算
        obs = self._compute_observation()
        reward, terminated = self._compute_reward(q, qdot, tau, action)
        
        # ステップ数管理
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        # ROS 2メッセージ配信
        self._publish_observation(obs)
        self._publish_reward(reward)
        self._publish_done(terminated or truncated)
        
        self.prev_action = action
        
        return obs, reward, terminated, truncated, {}
        
    def _compute_observation(self) -> np.ndarray:
        """観測ベクトルの計算"""
        q, qdot = self.sim.get_proprioception()
        base_orientation = self.sim.get_base_orientation()
        base_angular_vel = self.sim.get_base_angular_velocity()
        base_linear_acc = self.sim.get_base_linear_acceleration()
        
        # 観測正規化
        obs = np.concatenate([
            q / np.pi,  # 関節角度正規化
            np.clip(qdot / 10.0, -1.0, 1.0),  # 関節速度正規化
            base_orientation,  # ベース姿勢
            np.clip(base_angular_vel / 5.0, -1.0, 1.0),  # 角速度正規化
            np.clip(base_linear_acc / 20.0, -1.0, 1.0),  # 線形加速度正規化
            self.prev_action  # 前回アクション
        ])
        
        return obs.astype(np.float32)
        
    def _compute_reward(self, q: np.ndarray, qdot: np.ndarray, 
                       tau: np.ndarray, action: np.ndarray) -> Tuple[float, bool]:
        """報酬関数計算"""
        # 前進速度報酬
        forward_vel = self.sim.get_forward_velocity()
        target_vel = self.config['reward']['target_velocity']
        progress = -abs(forward_vel - target_vel)
        
        # エネルギー効率報酬
        energy = -np.sum(np.abs(tau * qdot)) * self.reward_weights['energy']
        
        # 姿勢安定性報酬
        orientation_error = np.linalg.norm(self.sim.get_base_orientation()[:2])
        stability = -orientation_error * self.reward_weights['stability']
        
        # 滑らかさ報酬
        smoothness = -np.sum(np.abs(action - self.prev_action)) * self.reward_weights['smoothness']
        
        # 関節制限ペナルティ
        joint_limits = self.sim.get_joint_limits_violation()
        joint_penalty = -joint_limits * self.reward_weights['joint_limits']
        
        # 転倒判定
        is_fallen = self.sim.is_fallen()
        if is_fallen:
            fall_penalty = self.reward_weights['fall']
        else:
            fall_penalty = 0.0
            
        total_reward = progress + energy + stability + smoothness + joint_penalty + fall_penalty
        
        return float(total_reward), is_fallen
        
    def _publish_observation(self, obs: np.ndarray) -> None:
        """ROS 2観測メッセージ配信"""
        msg = Float32MultiArray()
        msg.data = obs.tolist()
        self.obs_pub.publish(msg)
        
    def _publish_reward(self, reward: float) -> None:
        """ROS 2報酬メッセージ配信"""
        msg = Float32MultiArray()
        msg.data = [reward]
        self.reward_pub.publish(msg)
        
    def _publish_done(self, done: bool) -> None:
        """ROS 2終了メッセージ配信"""
        msg = Bool()
        msg.data = done
        self.done_pub.publish(msg)
        
    def close(self) -> None:
        """環境クリーンアップ"""
        self.node.destroy_node()
        self.sim.close()