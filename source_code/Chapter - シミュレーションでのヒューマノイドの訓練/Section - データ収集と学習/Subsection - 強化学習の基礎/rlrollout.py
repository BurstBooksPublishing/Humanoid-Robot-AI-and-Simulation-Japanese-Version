import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List
import numpy as np

class RolloutBuffer:
    """ロールアウトデータをGPU/CPUに連続で溜めておくバッファ"""
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.obs: List[Tensor] = []
        self.act: List[Tensor] = []
        self.rew: List[Tensor] = []
        self.done: List[Tensor] = []
        self.logp: List[Tensor] = []
        self.val: List[Tensor] = []

    def push(self, obs, act, rew, done, logp, val):
        self.obs.append(obs.detach().to(self.device))
        self.act.append(act.detach().to(self.device))
        self.rew.append(torch.as_tensor(rew, device=self.device))
        self.done.append(torch.as_tensor(done, device=self.device))
        self.logp.append(logp.detach().to(self.device))
        self.val.append(val.detach().to(self.device))

    def tensorize(self) -> Tuple[Tensor, ...]:
        return (torch.cat(self.obs),
                torch.cat(self.act),
                torch.cat(self.rew),
                torch.cat(self.done),
                torch.cat(self.logp),
                torch.cat(self.val))


@torch.no_grad()
def compute_gae(rewards: Tensor,
                values: Tensor,
                dones: Tensor,
                next_value: Tensor,
                gamma: float = 0.99,
                lam: float = 0.95) -> Tuple[Tensor, Tensor]:
    """GAE-λでadvantage/returnを計算（vectorized）"""
    T = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    gae = 0
    next_values = torch.cat([values[1:], next_value.unsqueeze(0)])
    deltas = rewards + gamma * next_values * (1 - dones) - values
    for t in reversed(range(T)):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    return returns, advantages


def ppo_update(policy: nn.Module,
               optimizer: torch.optim.Optimizer,
               buffer: RolloutBuffer,
               T: int,
               batch_size: int = 512,
               clip_eps: float = 0.2,
               epochs: int = 10,
               entropy_coef: float = 0.01,
               value_coef: float = 0.5,
               max_grad_norm: float = 1.0):
    """PPO-Clipで複数epoch更新"""
    obs, act, rew, done, logp_old, val_old = buffer.tensorize()
    with torch.no_grad():
        next_value = policy.critic(obs[-1]).squeeze(-1)
    returns, adv = compute_gae(rew, val_old, done, next_value)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    dataset = torch.utils.data.TensorDataset(obs, act, logp_old, returns, adv)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for o, a, l_old, ret, ad in loader:
            dist, val = policy(o)
            logp = dist.log_prob(a).sum(-1)
            ratio = torch.exp(logp - l_old)
            surr1 = ratio * ad
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * ad
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (ret - val.squeeze(-1)).pow(2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()