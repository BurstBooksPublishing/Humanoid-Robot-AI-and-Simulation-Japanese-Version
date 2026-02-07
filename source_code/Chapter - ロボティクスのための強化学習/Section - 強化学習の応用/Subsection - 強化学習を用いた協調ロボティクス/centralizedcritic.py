import torch
import torch.nn.functional as F
from typing import List, Tuple

def update_critics_and_policies(
    obs: List[torch.Tensor],
    actions: List[torch.Tensor],
    rewards: torch.Tensor,
    next_obs: List[torch.Tensor],
    dones: torch.Tensor,
    centralized_state: torch.Tensor,
    centralized_next_state: torch.Tensor,
    policies: List[torch.nn.Module],
    critic: torch.nn.Module,
    critic_optimizer: torch.optim.Optimizer,
    pi_optimizers: List[torch.optim.Optimizer],
    gamma: float,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    マルチエージェント集中クリティック・分散アクタ更新
    返り値: (critic_loss, actor_losses)
    """
    device = centralized_state.device
    batch_size = centralized_state.size(0)
    n_agents = len(policies)

    # --- クリティックターゲット計算 (TD(0)) ---
    with torch.no_grad():
        next_actions = []
        log_probs = []
        for i, pi in enumerate(policies):
            dist = pi(next_obs[i])
            act = dist.rsample()  # 再パラメータ化トリック
            next_actions.append(act)
            log_probs.append(dist.log_prob(act).sum(dim=-1, keepdim=True))

        q_next = critic(centralized_next_state, torch.cat(next_actions, dim=-1))
        target_q = rewards.sum(dim=1, keepdim=True) + gamma * (1.0 - dones.float()) * q_next

    # --- クリティック更新 ---
    q_pred = critic(centralized_state, torch.cat(actions, dim=-1))
    critic_loss = F.mse_loss(q_pred, target_q)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    # 勾配クリッピング
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
    critic_optimizer.step()

    # --- アクタ更新 (分散) ---
    actor_losses = []
    for i, pi in enumerate(policies):
        dist = pi(obs[i])
        sampled_action = dist.rsample()
        log_prob = dist.log_prob(sampled_action).sum(dim=-1, keepdim=True)

        all_actions = actions.copy()
        all_actions[i] = sampled_action
        q_val = critic(centralized_state, torch.cat(all_actions, dim=-1))

        # エントロピー正則化付き目的
        entropy = -log_prob
        actor_loss = -(q_val + 0.01 * entropy).mean()

        pi_optimizers[i].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), max_norm=1.0)
        pi_optimizers[i].step()
        actor_losses.append(actor_loss.detach())

    return critic_loss.detach(), actor_losses