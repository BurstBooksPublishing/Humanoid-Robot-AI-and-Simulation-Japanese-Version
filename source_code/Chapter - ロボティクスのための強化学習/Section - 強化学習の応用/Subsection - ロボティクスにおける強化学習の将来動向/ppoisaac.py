import os
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import rclpy
from rclpy.node import Node
from isaacgym import gymtorch, gymapi
from isaacgymenvs.tasks.base.vec_task import VecTask
from omegaconf import DictConfig
import hydra
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

class PolicyNetwork(nn.Module):
    """アクタ・クリティック方策ネットワーク"""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value = nn.Linear(hidden, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.backbone(x)
        mean = self.mean(feat)
        std = self.log_std.exp().expand_as(mean)
        value = self.value(feat).squeeze(-1)
        return mean, std, value

class RolloutBuffer:
    """ロールアウトデータ格納"""
    def __init__(self):
        self.obs, self.act, self.logp, self.val, self.rew, self.ret = [], [], [], [], [], []

    def push(self, obs, act, logp, val, rew):
        self.obs.append(obs)
        self.act.append(act)
        self.logp.append(logp)
        self.val.append(val)
        self.rew.append(rew)

    def compute_returns(self, gamma: float = 0.99, lam: float = 0.95, last_val: torch.Tensor = None):
        """GAE-Lambdaでリターン計算"""
        rewards = torch.cat(self.rew)
        values = torch.cat(self.val + [last_val])
        gae = 0
        returns = []
        for step in reversed(range(len(self.rew))):
            delta = rewards[step] + gamma * values[step + 1] - values[step]
            gae = delta + gamma * lam * gae
            returns.insert(0, gae + values[step])
        self.ret = returns

class PPOAgent(Node):
    def __init__(self, cfg: DictConfig):
        super().__init__('ppo_agent')
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.envs = hydra.utils.instantiate(cfg.task)
        self.policy = PolicyNetwork(
            self.envs.observation_space.shape[0],
            self.envs.action_space.shape[0]).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=1e-5)
        self.buffer = RolloutBuffer()
        self.epoch = 0
        self.eval_every = cfg.eval_every
        self.checkpoint_dir = cfg.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @torch.no_grad()
    def collect_rollouts(self) -> Dict[str, torch.Tensor]:
        """並列シムでバッチロールアウト収集"""
        obs = self.envs.reset()
        self.buffer = RolloutBuffer()
        for _ in range(self.cfg.steps_per_env):
            mean, std, val = self.policy(torch.from_numpy(obs).to(self.device))
            dist = Normal(mean, std)
            act = dist.sample()
            logp = dist.log_prob(act).sum(-1)
            next_obs, rew, done, info = self.envs.step(act.cpu().numpy())
            self.buffer.push(
                torch.from_numpy(obs).to(self.device),
                act,
                logp,
                val,
                torch.from_numpy(rew).to(self.device))
            obs = next_obs
        _, _, last_val = self.policy(torch.from_numpy(obs).to(self.device))
        self.buffer.compute_returns(self.cfg.gamma, self.cfg.lam, last_val)
        return {
            'obs': torch.cat(self.buffer.obs),
            'act': torch.cat(self.buffer.act),
            'logp': torch.cat(self.buffer.logp),
            'ret': torch.cat(self.buffer.ret),
            'val': torch.cat(self.buffer.val)}

    def ppo_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """PPOクリップ損失"""
        mean, std, val = self.policy(batch['obs'])
        dist = Normal(mean, std)
        new_logp = dist.log_prob(batch['act']).sum(-1)
        ratio = (new_logp - batch['logp']).exp()
        adv = (batch['ret'] - batch['val']).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = 0.5 * (batch['ret'] - val).pow(2).mean()
        entropy = dist.entropy().sum(-1).mean()
        return actor_loss + self.cfg.vf_coef * critic_loss - self.cfg.ent_coef * entropy

    def train_epoch(self):
        batch = self.collect_rollouts()
        for _ in range(self.cfg.epochs_per_update):
            idx = torch.randperm(len(batch['obs']))
            for start in range(0, len(batch['obs']), self.cfg.mini_batch):
                mb = {k: v[idx[start:start + self.cfg.mini_batch]] for k, v in batch.items()}
                loss = self.ppo_loss(mb)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
        self.epoch += 1

    def evaluate_on_robot(self):
        """短時間ハードウェアテスト（ROS 2サービス経由）"""
        client = self.create_client(Trigger, '/hardware/evaluate')
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('ハードウェア評価サービス未接続')
            return
        req = Trigger.Request()
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info(f'評価結果: {future.result()}')

    def save_checkpoint(self):
        ckpt = {
            'epoch': self.epoch,
            'model': self.policy.state_dict(),
            'optim': self.optimizer.state_dict()}
        path = os.path.join(self.checkpoint_dir, f'ckpt_{self.epoch:06d}.pt')
        torch.save(ckpt, path)
        torch.save(ckpt, os.path.join(self.checkpoint_dir, 'latest.pt'))

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    rclpy.init()
    agent = PPOAgent(cfg)
    for _ in range(cfg.total_epochs):
        agent.train_epoch()
        if agent.epoch % agent.eval_every == 0:
            agent.evaluate_on_robot()
            agent.save_checkpoint()
    rclpy.shutdown()

if __name__ == '__main__':
    main()