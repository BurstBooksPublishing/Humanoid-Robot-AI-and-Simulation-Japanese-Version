import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import rclpy
from rclpy.node import Node
from isaacsim import SimulationApp, Robot
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import yaml
import time
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IsaacSimEnv:
    """Isaac Sim環境ラッパー"""
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Isaac Sim初期化
        self.sim_app = SimulationApp({"headless": self.cfg["headless"]})
        self.robot = Robot(self.cfg["robot_usd_path"])
        self.stage = self.sim_app.get_stage()
        
        # 物理シーン設定
        self._setup_physics_scene()
        
    def _setup_physics_scene(self):
        """物理シーンの初期設定"""
        from pxr import UsdPhysics, PhysxSchema
        UsdPhysics.Scene.Define(self.stage, "/physicsScene")
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(self.stage.GetPrimAtPath("/physicsScene"))
        physxSceneAPI.CreateEnableCCDAttr().Set(True)
        
    def randomize_parameters(self, param_ranges: Dict[str, Tuple[float, float]]):
        """物理パラメータのランダマイゼーション"""
        for param, (min_val, max_val) in param_ranges.items():
            if param == "torso_mass":
                self.robot.set_body_mass("torso", np.random.uniform(min_val, max_val))
            elif param == "foot_friction":
                for foot in ["left_foot", "right_foot"]:
                    self.robot.set_material_friction(foot, np.random.uniform(min_val, max_val))
            elif param == "actuator_latency":
                self.robot.set_actuator_delay(np.random.uniform(min_val, max_val))
                
    def reset(self) -> np.ndarray:
        """環境リセット"""
        self.sim_app.reset()
        return self._get_observation()
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """1ステップ実行"""
        self.robot.apply_action(action)
        self.sim_app.step()
        
        obs = self._get_observation()
        reward = self._calculate_reward()
        done = self._check_termination()
        info = {}
        
        return obs, reward, done, info
        
    def _get_observation(self) -> np.ndarray:
        """観測値取得"""
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        base_ori = self.robot.get_base_orientation()
        base_ang_vel = self.robot.get_base_angular_velocity()
        
        return np.concatenate([joint_pos, joint_vel, base_ori, base_ang_vel])
        
    def _calculate_reward(self) -> float:
        """報酬計算"""
        velocity_reward = np.dot(self.robot.get_base_linear_velocity(), [1, 0, 0])
        energy_penalty = np.sum(np.square(self.robot.get_joint_efforts()))
        return velocity_reward - 0.001 * energy_penalty
        
    def _check_termination(self) -> bool:
        """終了条件チェック"""
        base_height = self.robot.get_base_position()[2]
        return base_height < 0.3  # 転倒判定

class HardwareLogger(Node):
    """実ハードウェアデータ収集ノード"""
    def __init__(self):
        super().__init__('hardware_logger')
        self.data_buffer = []
        
    def collect_trajectories(self, duration: float = 10.0) -> Dict:
        """実機データ収集"""
        start_time = time.time()
        while (time.time() - start_time) < duration:
            # ROS 2トピックからデータ取得
            joint_states = self._get_joint_states()
            imu_data = self._get_imu_data()
            
            self.data_buffer.append({
                'joint_positions': joint_states.position,
                'joint_velocities': joint_states.velocity,
                'imu_orientation': imu_data.orientation,
                'timestamp': time.time()
            })
            
        return {'trajectories': self.data_buffer}
        
    def _get_joint_states(self):
        """関節状態取得（ダミー実装）"""
        from sensor_msgs.msg import JointState
        return JointState()
        
    def _get_imu_data(self):
        """IMUデータ取得（ダミー実装）"""
        from sensor_msgs.msg import Imu
        return Imu()

def apply_action_filter(action: np.ndarray, cutoff_freq: float = 10.0) -> np.ndarray:
    """ローパスフィルタでアクションを平滑化"""
    # 簡単な指数移動平均フィルタ
    alpha = 2 * np.pi * cutoff_freq * 0.001  # 1msタイムステップ想定
    filtered_action = alpha * action + (1 - alpha) * action
    return filtered_action

def run_system_id(sim_env: IsaacSimEnv, real_data: Dict) -> Dict:
    """システム同定でシムパラメータを更新"""
    # 簡単な最小二乗法によるパラメータ推定
    sim_trajectory = collect_sim_trajectory(sim_env, len(real_data['trajectories']))
    
    # パラメータ誤差を計算
    param_error = {}
    for key in ['joint_positions', 'joint_velocities']:
        sim_vals = np.array([t[key] for t in sim_trajectory])
        real_vals = np.array([t[key] for t in real_data['trajectories']])
        param_error[key] = np.mean(np.square(sim_vals - real_vals))
        
    # 誤差に基づいてシムパラメータを調整
    correction_factor = 1.0 + 0.1 * np.tanh(param_error['joint_positions'])
    return {'mass_correction': correction_factor}

def collect_sim_trajectory(sim_env: IsaacSimEnv, num_steps: int) -> list:
    """シムから軌道データ収集"""
    trajectory = []
    obs = sim_env.reset()
    
    for _ in range(num_steps):
        action = np.zeros(sim_env.robot.num_dof)  # ゼロアクション
        obs, _, _, _ = sim_env.step(action)
        trajectory.append({
            'joint_positions': obs[:sim_env.robot.num_dof],
            'joint_velocities': obs[sim_env.robot.num_dof:2*sim_env.robot.num_dof]
        })
        
    return trajectory

def main():
    """メイン訓練ループ"""
    rclpy.init()
    
    # 環境初期化
    env = DummyVecEnv([lambda: IsaacSimEnv("config/isaac_cfg.yaml")])
    
    # PPOエージェント設定
    policy = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    # ハードウェアロガー初期化
    hw_logger = HardwareLogger()
    
    # 訓練ループ
    for episode in range(1000):
        # 物理パラメータランダマイゼーション
        env.env_method(
            "randomize_parameters",
            {
                "torso_mass": (8.0, 12.0),
                "foot_friction": (0.6, 1.2),
                "actuator_latency": (0.005, 0.02),
            }
        )
        
        # PPO更新
        policy.learn(total_timesteps=2048)
        
        # 定期的に実機データでキャリブレーション
        if episode % 50 == 0:
            real_data = hw_logger.collect_trajectories(duration=5.0)
            phi = run_system_id(env.envs[0], real_data)
            logger.info(f"Episode {episode}: System ID correction = {phi}")
            
    rclpy.shutdown()

if __name__ == "__main__":
    main()