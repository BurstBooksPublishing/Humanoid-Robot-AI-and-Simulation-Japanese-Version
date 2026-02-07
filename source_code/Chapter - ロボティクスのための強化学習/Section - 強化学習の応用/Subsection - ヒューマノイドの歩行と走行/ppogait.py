import numpy as np
import torch
from typing import Tuple, Dict, Any

# 環境インターフェース（型ヒント付き）
class Env:
    def reset(self) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]: ...

# バッファ（Tensor化＋GPU対応）
class RolloutBuffer:
    def __init__(self, device: torch.device):
        self.device = device
        self.clear()

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, done: bool):
        self.obs.append(torch.from_numpy(obs).float())
        self.actions.append(torch.from_numpy(act).float())
        self.rewards.append(rew)
        self.dones.append(done)

    def tensor(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (torch.stack(self.obs).to(self.device),
                torch.stack(self.actions).to(self.device),
                torch.tensor(self.rewards, device=self.device),
                torch.tensor(self.dones, device=self.device))

    def clear(self):
        self.obs, self.actions, self.rewards, self.dones = [], [], [], []

# GAE計算（バッチ化）
@torch.no_grad()
def compute_gae(rewards: torch.Tensor,
                dones: torch.Tensor,
                values: torch.Tensor,
                gamma: float = 0.99,
                lam: float = 0.95) -> torch.Tensor:
    T = len(rewards)
    gae = 0
    adv = torch.zeros_like(rewards)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    return adv

# 本番コード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
buffer = RolloutBuffer(device)

for episode in range(num_episodes):
    obs = env.reset()
    v_target = sample_velocity()
    policy.set_command(v_target)

    done = False
    while not done:
        action = policy.act(torch.from_numpy(obs).float().to(device))
        next_obs, r, done, info = env.step(action.cpu().numpy())
        buffer.store(obs, action.cpu().numpy(), r, done)
        obs = next_obs

    obs_t, act_t, rew_t, done_t = buffer.tensor()
    values = policy.critic(obs_t).squeeze()
    advantages = compute_gae(rew_t, done_t, torch.cat([values, values[-1:]]))
    policy.update(obs_t, act_t, advantages)
    buffer.clear()