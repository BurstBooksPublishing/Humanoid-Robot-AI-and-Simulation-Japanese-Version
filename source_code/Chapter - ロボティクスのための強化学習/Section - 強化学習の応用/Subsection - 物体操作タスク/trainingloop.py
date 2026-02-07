#!/usr/bin/env python3
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# ROS 2
import rclpy
from rclpy.node import Node

# Isaac Sim
from omni.isaac.kit import SimulationApp
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import TactileSensor
from omni.isaac.core.utils.types import ArticulationAction

# RL
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# MPC (for safety filter)
from do_mpc.controller import MPC
from do_mpc.model import Model
from do_mpc.tools import structure

# 設定
@dataclass
class TrainConfig:
    exp_name: str = "isaac_ppo_tactile"
    num_episodes: int = 5000
    eval_interval: int = 50
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "./runs"
    save_dir: str = "./checkpoints"
    isaac_usd_path: str = "/Isaac/Samples/Tasks/PickAndPlace/pick_and_place.usd"
    tactile_prim_path: str = "/World/robot/gripper/tactile"
    robot_prim_path: str = "/World/robot"
    camera_prim_path: str = "/World/camera"
    mpc_horizon: int = 10
    mpc_dt: float = 0.1


# Isaac Sim 環境
class IsaacSimEnv(Node):
    def __init__(self, cfg: TrainConfig, headless: bool = True) -> None:
        Node.__init__(self, "isaac_env")
        self.cfg = cfg
        self.headless = headless
        self._init_sim()
        self._init_scene()
        self._init_sensors()
        self._init_spaces()

    def _init_sim(self) -> None:
        # Isaac Sim起動
        self.simulation_app = SimulationApp({"headless": self.headless})
        enable_extension("omni.isaac.sensor")
        self.simulation_context = SimulationContext(
            physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0, stage_units_in_meters=1.0
        )

    def _init_scene(self) -> None:
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + self.cfg.isaac_usd_path
        open_stage(usd_path)
        self.scene = Scene()
        self.scene.add_default_ground_plane()
        set_camera_view(eye=[2, 2, 2], target=[0, 0, 0])

        # ロボット読み込み
        define_prim(self.cfg.robot_prim_path, "Xform")
        self.robot = XFormPrimView(self.cfg.robot_prim_path)

    def _init_sensors(self) -> None:
        # 触覚センサ
        self.tactile = TactileSensor(
            prim_path=self.cfg.tactile_prim_path,
            name="tactile",
            translation=np.array([0, 0, 0.01]),
        )
        self.tactile.initialize()

    def _init_spaces(self) -> None:
        # 観測・行動空間定義
        self.obs_dim = 3 + 3 + 48  # 画像特徴 + 関節 + 触覚
        self.act_dim = 6  # エンドエフェクタ速度
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        self.simulation_context.reset()
        self.tactile.reset()
        obs = self._get_obs()
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # 安全フィルタ適用 (MPC)
        safe_action = self._safety_filter(action)
        self.robot.apply_action(
            ArticulationAction(joint_velocities=safe_action)
        )
        self.simulation_context.step(render=not self.headless)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}
        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        # 画像特徴抽出 (簡易CNN)
        img = self._get_camera_image()
        img_feat = self._encode_image(img)
        joints = self.robot.get_joint_positions()
        tactile_vals = self.tactile.get_sensor_values()
        obs = np.concatenate([img_feat, joints, tactile_vals])
        return obs.astype(np.float32)

    def _encode_image(self, img: np.ndarray) -> np.ndarray:
        # 軽量CNN (推論用)
        if not hasattr(self, "_img_encoder"):
            self._img_encoder = (
                nn.Sequential(
                    nn.Conv2d(3, 16, 3, 2),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(16 * 32 * 32, 3),
                )
                .to(self.cfg.device)
                .eval()
            )
        with torch.no_grad():
            x = torch.from_numpy(img).unsqueeze(0).to(self.cfg.device)
            feat = self._img_encoder(x).squeeze(0).cpu().numpy()
        return feat

    def _get_camera_image(self) -> np.ndarray:
        # カメラ画像取得 (256x256x3)
        import omni.isaac.synthetic_utils as su
        viewport = su.get_viewport_from_camera(self.cfg.camera_prim_path)
        return su.get_rgb(viewport)

    def _compute_reward(self) -> float:
        # 報酬はタスク依存 (簡易)
        return 1.0

    def _check_done(self) -> bool:
        # 終了条件
        return False

    def _safety_filter(self, action: np.ndarray) -> np.ndarray:
        # MPCベース安全フィルタ
        if not hasattr(self, "_mpc"):
            self._setup_mpc()
        x0 = self._get_state()
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
        u = self.mpc.make_step(x0)
        return u.flatten()

    def _setup_mpc(self) -> None:
        model_type = "continuous"
        model = Model(model_type)
        # 状態: 関節位置・速度
        model.set_variable("_x", "q", (6, 1))
        model.set_variable("_x", "dq", (6, 1))
        # 入力: 関節速度
        model.set_variable("_u", "u", (6, 1))
        # ダイナミクス
        model.set_rhs("q", model.x["dq"])
        model.set_rhs("dq", model.u["u"])
        model.setup()
        self.mpc = MPC(model)
        setup_mpc = {
            "n_horizon": self.cfg.mpc_horizon,
            "t_step": self.cfg.mpc_dt,
            "n_robust": 0,
            "store_full_solution": False,
        }
        self.mpc.set_param(**setup_mpc)
        # 制約
        self.mpc.set_rterm(u=1e-2)
        self.mpc.setup()

    def _get_state(self) -> np.ndarray:
        q = self.robot.get_joint_positions()
        dq = self.robot.get_joint_velocities()
        return np.concatenate([q, dq])


# 評価用コールバック
class IsaacEvalCallback(EvalCallback):
    def __init__(self, eval_env, **kwargs):
        super().__init__(eval_env, **kwargs)

    def _on_step(self) -> bool:
        # 安全メトリクス記録
        return super()._on_step()


# メイン
def main():
    cfg = TrainConfig()
    set_random_seed(cfg.seed)

    # ログ・保存ディレクトリ
    Path(cfg.log_dir).mkdir(exist_ok=True)
    Path(cfg.save_dir).mkdir(exist_ok=True)

    # 環境生成
    rclpy.init()
    env = IsaacSimEnv(cfg, headless=True)
    check_env(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # 評価環境
    eval_env = IsaacSimEnv(cfg, headless=True)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    # PPOモデル
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=cfg.log_dir,
        device=cfg.device,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        ent_coef=0.0,
        vf_coef=0.5,
    )

    # コールバック
    eval_callback = IsaacEvalCallback(
        eval_env,
        best_model_save_path=cfg.save_dir,
        log_path=cfg.log_dir,
        eval_freq=cfg.eval_interval * env.num_envs * 2048,
        deterministic=True,
        render=False,
    )
    ckpt_callback = CheckpointCallback(
        save_freq=100 * env.num_envs * 2048,
        save_path=cfg.save_dir,
        name_prefix="model",
    )

    # 学習
    model.learn(
        total_timesteps=cfg.num_episodes * 2048,
        callback=[eval_callback, ckpt_callback],
    )

    # 保存
    model.save(os.path.join(cfg.save_dir, "final_model"))
    env.save(os.path.join(cfg.save_dir, "vec_normalize.pkl"))

    # 終了
    env.close()
    eval_env.close()
    rclpy.shutdown()
    env.simulation_app.close()


if __name__ == "__main__":
    main()