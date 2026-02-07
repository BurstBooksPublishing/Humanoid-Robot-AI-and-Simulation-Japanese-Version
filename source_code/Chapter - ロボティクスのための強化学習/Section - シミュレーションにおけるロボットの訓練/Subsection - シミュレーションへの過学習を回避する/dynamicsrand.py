import numpy as np
import torch
from typing import Dict, Any
import rclpy
from rclpy.node import Node
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.core.simulation_context import SimulationContext
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
import yaml
import time
import logging

class DomainRandomizer:
    """Isaac Simの物理パラメータをランダマイズ"""
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.param_ranges = {
            "mass_scale": (0.9, 1.2),
            "friction": (0.4, 1.2),
            "joint_damping": (0.7, 1.3),
            "action_delay": (0, 3),
            "imu_noise_std": (0.01, 0.08),
        }
    
    def sample(self) -> Dict[str, Any]:
        return {k: self.rng.uniform(*v) if k != "action_delay" else self.rng.integers(*v)
                for k, v in self.param_ranges.items()}

class IsaacEnvWrapper:
    """Isaac Sim環境のラッパー"""
    def __init__(self, stage_path: str, robot_prim: str):
        self.sim = SimulationContext()
        define_prim(stage_path, "Xform")
        self.robot_prim = robot_prim
        self.dr = DomainRandomizer(np.random.default_rng())
        
    def apply_physics_params(self, params: Dict[str, Any]):
        """USDプロパティに物理パラメータを反映"""
        from pxr import UsdPhysics, PhysxSchema
        stage = self.sim.stage
        for prim in stage.Traverse():
            if prim.GetTypeName() == "PhysicsRigidBodyAPI":
                mass_api = UsdPhysics.MassAPI(prim)
                if mass_api:
                    base_mass = mass_api.GetMassAttr().Get()
                    mass_api.GetMassAttr().Set(base_mass * params["mass_scale"])
            
            if prim.GetTypeName() == "PhysicsMaterial":
                material = UsdPhysics.MaterialAPI(prim)
                material.GetStaticFrictionAttr().Set(params["friction"])
                material.GetDynamicFrictionAttr().Set(params["friction"])
    
    def reset(self):
        self.sim.reset()
        return self._get_obs()
    
    def step(self, action):
        self.sim.step()
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self._check_done(obs)
        return obs, reward, done, {}
    
    def _get_obs(self):
        """実装に応じて観測を取得"""
        return np.zeros(42)  # dummy
    
    def _compute_reward(self, obs, action):
        return -np.sum(action**2)
    
    def _check_done(self, obs):
        return False

class DelayedActionWrapper:
    """アクション遅延をエミュレート"""
    def __init__(self, delay_steps: int):
        self.delay_steps = delay_steps
        self.queue = []
    
    def __call__(self, action: np.ndarray) -> np.ndarray:
        self.queue.append(action.copy())
        if len(self.queue) > self.delay_steps:
            return self.queue.pop(0)
        return np.zeros_like(action)

def main():
    rclpy.init()
    node = Node("isaac_rl_train")
    
    # 設定読み込み
    with open("config/train_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    env = IsaacEnvWrapper("/World/robot", "/World/robot/torso")
    replay_buffer = ReplayBuffer(
        cfg["buffer_size"],
        env.observation_space,
        env.action_space,
        device="cuda",
        n_envs=1
    )
    
    policy = SAC("MlpPolicy", env, verbose=1, device="cuda")
    
    for episode in range(cfg["num_episodes"]):
        params = env.dr.sample()
        env.apply_physics_params(params)
        
        obs = env.reset()
        delay_wrapper = DelayedActionWrapper(int(params["action_delay"]))
        
        done = False
        episode_reward = 0.0
        
        while not done:
            action, _ = policy.predict(obs, deterministic=False)
            action = delay_wrapper(action)
            
            obs, reward, done, info = env.step(action)
            replay_buffer.add(obs, action, reward, done, info)
            episode_reward += reward
            
            if replay_buffer.size() > cfg["learning_starts"]:
                policy.train(gradient_steps=cfg["gradient_steps"])
        
        node.get_logger().info(f"Episode {episode}: reward={episode_reward:.2f}")
    
    policy.save("isaac_sac_policy")
    rclpy.shutdown()

if __name__ == "__main__":
    main()