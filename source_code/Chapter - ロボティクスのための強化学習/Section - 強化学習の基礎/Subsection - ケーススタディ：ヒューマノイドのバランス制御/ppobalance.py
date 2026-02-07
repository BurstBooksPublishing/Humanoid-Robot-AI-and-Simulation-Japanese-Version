import os
from typing import Any, Dict, Tuple

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class IsaacHumanoidRewardWrapper(gym.Wrapper):
    """Isaac環境の報酬を上書きし、直立・省エネ・滑らかさを考慮"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_action: np.ndarray = np.zeros(env.action_space.shape)

    def reset(self, **kwargs) -> Dict[str, np.ndarray]:
        obs = self.env.reset(**kwargs)
        self._prev_action = np.zeros(self.env.action_space.shape)
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        obs, _, done, info = self.env.step(action)

        # 直立性：ロール・ピッチのノルムをペナルティ
        theta = obs["torso_angles"][:2]
        upright = 5.0 * np.exp(-2.0 * np.linalg.norm(theta))

        # エネルギー効率：トルク×速度
        energy = -0.01 * np.sum(np.abs(action * obs["joint_vel"]))

        # ジャーク：行動変化を滑らかさの代理
        jerk = -0.005 * np.sum(np.abs(action - self._prev_action))

        # 転倒
        fall = -50.0 if obs.get("fallen", False) else 0.0

        reward = float(upright + energy + jerk + fall)

        self._prev_action = action.copy()
        return obs, reward, done, info


def make_env(rank: int = 0) -> gym.Env:
    """環境生成：ラップ＋監視＋ベクトル化用"""
    def _init() -> gym.Env:
        env = gym.make("IsaacHumanoidBalance-v0")
        env = IsaacHumanoidRewardWrapper(env)
        env = Monitor(env, filename=None)  # 統計記録
        return env

    return _init


if __name__ == "__main__":
    # 並列環境（ここでは1つ）
    n_envs = 1
    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # 保存先
    log_dir = "./runs/ppo_balance"
    os.makedirs(log_dir, exist_ok=True)

    # コールバック：定期的に評価＆チェックポイント保存
    eval_env = DummyVecEnv([make_env(999)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=50_000 // n_envs,
        deterministic=True,
        render=False,
    )
    ckpt_callback = CheckpointCallback(
        save_freq=200_000 // n_envs,
        save_path=log_dir,
        name_prefix="ppo_humanoid",
    )

    # 学習
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
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
    model.learn(total_timesteps=5_000_000, callback=[eval_callback, ckpt_callback])

    # 保存
    model.save(os.path.join(log_dir, "final_model"))
    vec_env.save(os.path.join(log_dir, "vec_normalize.pkl"))