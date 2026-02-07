import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# モック通信API（実環境では差し替え）
async def send_broadcast(payload: str) -> None:
    """非公開の通信層へ転送（本番ではDDS／ROS 2等に置換）"""
    pass

async def recv_broadcast() -> str:
    """非公開の通信層から受信（本番ではDDS／ROS 2等に置換）"""
    pass


@dataclass
class Config:
    agent_id: str = "humanoid_A"
    bid_ttl: float = 1.5
    heartbeat: float = 0.2
    max_queue_size: int = 1000


@dataclass
class Bid:
    agent: str
    cost: float
    ts: float


@dataclass
class TaskBoard:
    queues: Dict[str, List[Bid]] = field(default_factory=lambda: defaultdict(list))

    def prune(self, ttl: float) -> None:
        now = time.time()
        for bids in self.queues.values():
            bids[:] = [b for b in bids if now - b.ts < ttl]

    def winner(self, task: str) -> Optional[str]:
        bids = self.queues.get(task, [])
        if not bids:
            return None
        return min(bids, key=lambda b: b.cost).agent


class AuctionAgent:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.board = TaskBoard()
        self.logger = logging.getLogger(self.cfg.agent_id)

    async def send_bid(self, task_id: str, cost: float) -> None:
        msg = {
            "type": "bid",
            "agent": self.cfg.agent_id,
            "task": task_id,
            "cost": cost,
            "t": time.time(),
        }
        await send_broadcast(json.dumps(msg))

    async def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("受信データがJSONではありません")
            return

        if msg.get("type") == "bid":
            task = msg.get("task")
            if task is None:
                return
            self.board.queues[task].append(
                Bid(agent=msg["agent"], cost=msg["cost"], ts=msg["t"])
            )
            # メモリ保護
            if len(self.board.queues[task]) > self.cfg.max_queue_size:
                self.board.queues[task].pop(0)

    async def listen_loop(self) -> None:
        while True:
            raw = await recv_broadcast()
            await self._handle_message(raw)

    async def auction_loop(self) -> None:
        while True:
            self.board.prune(self.cfg.bid_ttl)

            for task in list(self.board.queues.keys()):
                winner = self.board.winner(task)
                if winner is None:
                    continue
                await send_broadcast(
                    json.dumps({"type": "award", "task": task, "agent": winner})
                )
                del self.board.queues[task]

            await asyncio.sleep(self.cfg.heartbeat)

    async def run(self) -> None:
        await asyncio.gather(self.listen_loop(), self.auction_loop())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = Config()
    agent = AuctionAgent(cfg)
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        pass