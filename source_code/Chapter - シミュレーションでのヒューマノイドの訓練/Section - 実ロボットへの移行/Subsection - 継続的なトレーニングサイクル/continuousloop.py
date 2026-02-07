#!/usr/bin/env python3
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader

# ROS 2 (オプション)
try:
    import rclpy
    from rclpy.node import Node
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# ロギング
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定数
DIVERGENCE_THRESHOLD = 0.25
MAX_REPLAY_SIZE = 100_000
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ構造
@dataclass
class SensorCommand:
    timestamp: float
    sensor: np.ndarray
    command: np.ndarray

@dataclass
class Batch:
    seed: int
    data: List[SensorCommand]
    stats: dict

    @property
    def emb(self) -> torch.Tensor:
        # 簡易埋め込み：sensorとcommandを連結
        return torch.tensor(np.vstack([np.hstack([d.sensor, d.command]) for d in self.data]),
                            dtype=torch.float32, device=DEVICE)

class ReplayBuffer(TorchDataset):
    def __init__(self, max_size: int = MAX_REPLAY_SIZE):
        self.buffer: List[Tuple[Batch, Batch]] = []
        self.max_size = max_size

    def add(self, real: Batch, sim: Batch):
        self.buffer.append((real, sim))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        real, sim = self.buffer[idx]
        return real.emb, sim.emb

    def sample(self, batch_size: int = BATCH_SIZE):
        real_embs, sim_embs = zip(*random.sample(self.buffer, batch_size))
        return torch.stack(real_embs), torch.stack(sim_embs)

# モデル
class DivergenceEstimator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ロボットインターフェース
class RealRobot:
    def __init__(self, node_name: str = "real_robot_node"):
        if ROS2_AVAILABLE:
            rclpy.init()
            self.node = Node(node_name)
        else:
            self.node = None

    def record(self, duration: float) -> Batch:
        # 実機からセンサ＋指令を収集
        data = []
        start = time.time()
        while (time.time() - start) < duration:
            # ダミー読み取り
            sensor = np.random.randn(10)
            command = np.random.randn(5)
            data.append(SensorCommand(time.time(), sensor, command))
            time.sleep(0.01)
        stats = {"mean_sensor": np.mean([d.sensor for d in data], axis=0)}
        seed = random.randint(0, 2**31 - 1)
        return Batch(seed=seed, data=data, stats=stats)

# シミュレータ
class Simulator:
    def __init__(self):
        self.params = {"friction": 0.5, "delay": 0.0}

    def generate(self, seed: int) -> Batch:
        # シード固定で再現性確保
        random.seed(seed)
        np.random.seed(seed)
        data = []
        for _ in range(6000):  # 60秒分 @ 100Hz
            sensor = np.random.randn(10) * self.params["friction"]
            command = np.random.randn(5) * (1 + self.params["delay"])
            data.append(SensorCommand(time.time(), sensor, command))
        stats = {"mean_sensor": np.mean([d.sensor for d in data], axis=0)}
        return Batch(seed=seed, data=data, stats=stats)

    def adapt(self, real_stats: dict):
        # 摩擦・遅延を更新
        self.params["friction"] *= np.random.uniform(0.9, 1.1)
        self.params["delay"] = max(0.0, self.params["delay"] + np.random.uniform(-0.01, 0.01))

# 損失関数
def compute_divergence(real_emb: torch.Tensor, sim_emb: torch.Tensor) -> float:
    # MMD簡易実装
    xx = torch.mean(torch.mm(real_emb, real_emb.T))
    yy = torch.mean(torch.mm(sim_emb, sim_emb.T))
    xy = torch.mean(torch.mm(real_emb, sim_emb.T))
    mmd = xx + yy - 2 * xy
    return mmd.item()

# ファインチューニング
def fine_tune(model: nn.Module, dataset: ReplayBuffer, lr: float = 1e-4, steps: int = 1000) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(steps // len(loader) + 1):
        for real_emb, sim_emb in loader:
            optimizer.zero_grad()
            loss = compute_divergence(real_emb, sim_emb)
            torch.tensor(loss).backward()
            optimizer.step()
    return model

# 安全検証
def safety_validate(model: nn.Module, shadow_logs: List[Batch], thresholds: dict) -> bool:
    for log in shadow_logs:
        sim_log = Simulator().generate(seed=log.seed)
        div = compute_divergence(log.emb, sim_log.emb)
        if div > thresholds.get("max_div", 0.3):
            return False
    return True

# デプロイ
def deploy(model: nn.Module):
    logger.info("モデルを本番環境へデプロイ")
    torch.save(model.state_dict(), "/tmp/deployed_model.pt")

# メイン
def main():
    robot = RealRobot()
    sim = Simulator()
    dataset = ReplayBuffer()
    model = DivergenceEstimator(input_dim=15)
    shadow_logs = [robot.record(10) for _ in range(5)]  # シャドウ用ログ

    while True:
        real_batch = robot.record(duration=60)
        sim_batch = sim.generate(seed=real_batch.seed)
        div = compute_divergence(real_batch.emb, sim_batch.emb)
        if div > DIVERGENCE_THRESHOLD:
            sim.adapt(real_batch.stats)
        dataset.add(real_batch, sim_batch)
        model = fine_tune(model, dataset)
        if safety_validate(model, shadow_logs, {"max_div": 0.3}):
            deploy(model)
        time.sleep(1)  # 次サイクルまで待機

if __name__ == "__main__":
    main()