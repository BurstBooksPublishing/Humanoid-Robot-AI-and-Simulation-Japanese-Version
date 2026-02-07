import os
import time
import signal
import logging
from queue import Queue, Empty
from threading import Thread, Event
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 環境変数でバッチサイズ・デバイスを上書き可能
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
DEVICE = os.getenv("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# シャットダウン用グローバルイベント
shutdown_event = Event()


@dataclass
class InferenceItem:
    """推論リクエスト1件"""
    obs_id: int
    obs: torch.Tensor


class PolicyModel(nn.Module):
    """本番用は外部ファイルからロード"""
    def __init__(self, ckpt_path: str):
        super().__init__()
        # 例：torch.load(ckpt_path, map_location="cpu")
        # self.backbone = ...
        # self.head = ...
        logger.info(f"Model loaded from {ckpt_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 実装省略
        return x


class BatchInferenceService:
    def __init__(
        self,
        model: nn.Module,
        device: str,
        batch_size: int,
        max_delay: float = 0.01,
        num_threads: int = 1,
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.max_delay = max_delay  # バッチが埋まらない場合の最大待機[s]

        self.in_queue: Queue[InferenceItem] = Queue()
        self.out_queue: Queue[Tuple[int, torch.Tensor]] = Queue()

        self.threads: List[Thread] = []
        for _ in range(num_threads):
            t = Thread(target=self._worker, daemon=True)
            t.start()
            self.threads.append(t)

    def _worker(self):
        """バッチ推論ワーカー"""
        while not shutdown_event.is_set():
            batch: List[InferenceItem] = []
            deadline = time.time() + self.max_delay

            # バッチを溜める
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    item = self.in_queue.get(timeout=0.001)
                    batch.append(item)
                except Empty:
                    continue

            if not batch:
                continue

            # 推論
            ids = [it.obs_id for it in batch]
            x = torch.stack([it.obs for it in batch]).to(self.device)
            with torch.no_grad(), autocast(enabled=self.device.startswith("cuda")):
                actions = self.model(x)

            # 結果を返す
            for idx, act in zip(ids, actions.cpu()):
                self.out_queue.put((idx, act))

    def submit(self, obs_id: int, obs: torch.Tensor):
        self.in_queue.put(InferenceItem(obs_id, obs))

    def get(self, timeout: Optional[float] = None) -> Tuple[int, torch.Tensor]:
        return self.out_queue.get(timeout=timeout)

    def stop(self):
        shutdown_event.set()
        for t in self.threads:
            t.join()


# グレースフルシャットダウン
def _signal_handler(signum, frame):
    logger.info("Shutdown signal received")
    shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# 初期化
model = PolicyModel("policy.ckpt")
service = BatchInferenceService(model, DEVICE, BATCH_SIZE)

# 使用例
if __name__ == "__main__":
    # シミュレータ側
    dummy_obs = torch.randn(4, 84, 84)
    service.submit(agent_id=0, obs=dummy_obs)

    # 制御側
    aid, action = service.get(timeout=0.1)
    logger.info(f"Agent {aid} -> action {action}")