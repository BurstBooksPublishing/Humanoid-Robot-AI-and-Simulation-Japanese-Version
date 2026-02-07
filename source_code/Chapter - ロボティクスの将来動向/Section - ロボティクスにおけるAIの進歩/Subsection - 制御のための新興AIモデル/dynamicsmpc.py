import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class DynamicsModel(nn.Module):
    """
    確定的ダイナミクスモデル：s_{t+1} = s_t + f(s_t, a_t)
    正規化・デノイズ機能を内蔵し，安定学習を支援
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 512,
                 drop_prob: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 入力統計量（推論時に使用）
        self.register_buffer('s_mean', torch.zeros(state_dim))
        self.register_buffer('s_std', torch.ones(state_dim))
        self.register_buffer('a_mean', torch.zeros(action_dim))
        self.register_buffer('a_std', torch.ones(action_dim))
        self.register_buffer('ds_mean', torch.zeros(state_dim))
        self.register_buffer('ds_std', torch.ones(state_dim))

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # 正規化入力
        s_norm = (state - self.s_mean) / (self.s_std + 1e-6)
        a_norm = (action - self.a_mean) / (self.a_std + 1e-6)
        x = torch.cat([s_norm, a_norm], dim=-1)
        ds_norm = self.net(x)
        # デ正規化して残差を加算
        ds = ds_norm * self.ds_std + self.ds_mean
        return state + ds

    def update_stats(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     deltas: torch.Tensor):
        """データセット全体で統計量を更新"""
        self.s_mean.copy_(states.mean(0))
        self.s_std.copy_(states.std(0))
        self.a_mean.copy_(actions.mean(0))
        self.a_std.copy_(actions.std(0))
        self.ds_mean.copy_(deltas.mean(0))
        self.ds_std.copy_(deltas.std(0))


class ModelTrainer:
    def __init__(self,
                 model: DynamicsModel,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=10, factor=0.5, verbose=False)

    def train_epoch(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    next_states: torch.Tensor) -> float:
        self.model.train()
        states, actions, next_states = [x.to(self.device) for x in
                                        (states, actions, next_states)]
        deltas = next_states - states
        pred = self.model(states, actions)
        loss = F.mse_loss(pred, next_states)

        self.opt.zero_grad()
        loss.backward()
        # 勾配クリップ
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.opt.step()
        return loss.item()

    def fit(self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            epochs: int = 500,
            batch_size: int = 256,
            val_split: float = 0.1) -> None:
        n = states.size(0)
        n_val = int(n * val_split)
        perm = torch.randperm(n)
        train_idx, val_idx = perm[n_val:], perm[:n_val]

        ds = next_states - states
        self.model.update_stats(states, actions, ds)

        best_val = float('inf')
        for epoch in range(epochs):
            # ミニバッチ学習
            batch_idx = train_idx[torch.randperm(len(train_idx))[:batch_size]]
            tr_loss = self.train_epoch(states[batch_idx],
                                       actions[batch_idx],
                                       next_states[batch_idx])

            # 検証
            with torch.no_grad():
                val_pred = self.model(states[val_idx], actions[val_idx])
                val_loss = F.mse_loss(val_pred, next_states[val_idx]).item()

            self.scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
            else:
                if self.opt.param_groups[0]['lr'] < 1e-6:
                    break


class ShootingMPC:
    """
    シンプルなランダムシャーティングMPC
    カーネル融合版コスト計算を含む
    """
    def __init__(self,
                 model: DynamicsModel,
                 cost_fn,
                 horizon: int = 20,
                 num_samples: int = 1024,
                 device: str = 'cuda'):
        self.model = model
        self.cost_fn = cost_fn
        self.horizon = horizon
        self.num_samples = num_samples
        self.device = device
        self.action_bounds = (-1.0, 1.0)  # 正規化済みアクション範囲

    def __call__(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x0 = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            # 一様ランダムアクション候補
            A = torch.empty(self.num_samples, self.horizon, self.model.action_dim,
                            device=self.device).uniform_(*self.action_bounds)
            X = x0.repeat(self.num_samples, 1)
            total_cost = torch.zeros(self.num_samples, device=self.device)

            for t in range(self.horizon):
                a = A[:, t, :]
                X = self.model(X, a)
                total_cost += self.cost_fn(X, a)

            best_idx = torch.argmin(total_cost)
            best_act = A[best_idx, 0, :].cpu().numpy()
            return best_act


# ------------------------------------------------------------------
# 使用例（別ファイルで定義するcost_fnを渡す）
# ------------------------------------------------------------------
if __name__ == "__main__":
    # ダミーデータ
    states = torch.randn(10000, 50)
    actions = torch.randn(10000, 12)
    next_states = states + torch.randn(10000, 50) * 0.1

    model = DynamicsModel(state_dim=50, action_dim=12)
    trainer = ModelTrainer(model)
    trainer.fit(states, actions, next_states)

    # コスト関数例：状態ノルム＋アクション正則化
    def cost_fn(x, a):
        return x.norm(dim=-1) + 1e-3 * a.norm(dim=-1)

    mpc = ShootingMPC(model, cost_fn)
    x0 = np.zeros(50)
    u = mpc(x0)