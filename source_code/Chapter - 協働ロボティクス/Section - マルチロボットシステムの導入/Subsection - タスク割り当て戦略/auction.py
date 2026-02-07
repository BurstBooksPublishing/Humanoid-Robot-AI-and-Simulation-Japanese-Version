import time
import logging
from typing import Dict, List, Tuple, Any, Optional
import heapq
import threading
from dataclasses import dataclass
from collections import defaultdict

# 入札メッセージ構造
@dataclass(frozen=True)
class Bid:
    robot: str
    task: str
    cost: float
    ts: float = time.time()

# 割当結果
@dataclass
class Assignment:
    task: str
    robot: str
    cost: float

class Auctioneer:
    """
    分散型タスク割当オークションノード。
    同一ロボットが複数回呼ばれることもあるため、スレッドセーフに実装。
    """
    def __init__(self, robot_id: str, comm, cap_check=None, cost_fn=None):
        self.robot_id = robot_id
        self.comm = comm
        self.cap_check = cap_check or (lambda r, t: True)
        self.cost_fn = cost_fn or (lambda r, t: 0.0)
        self._lock = threading.Lock()
        self._bids: Dict[str, List[Bid]] = defaultdict(list)  # task -> [Bid, ...]
        self.logger = logging.getLogger(f"Auctioneer[{robot_id}]")

    # 1ラウンドの入札・割当
    def auction_round(self, tasks: List[str], timeout: float = 0.2) -> List[Assignment]:
        # 自身の入札を送信
        my_bids = []
        for t in tasks:
            if not self.cap_check(self.robot_id, t):
                continue
            c = self.cost_fn(self.robot_id, t)
            bid = Bid(self.robot_id, t, c)
            my_bids.append(bid)
            self.comm.broadcast(bid.__dict__)  # 非ブロッキング

        # 受信スレッドで他ロボットの入札を収集
        stop_at = time.time() + timeout
        while time.time() < stop_at:
            msg = self.comm.recv(timeout=stop_at - time.time())
            if msg is None:
                continue
            if msg.get("type") != "bid":
                continue
            b = Bid(msg["robot"], msg["task"], msg["cost"], msg.get("ts", time.time()))
            with self._lock:
                heapq.heappush(self._bids[b.task], b)

        # 各タスクの最低コスト入札を選択
        winners: List[Assignment] = []
        with self._lock:
            for t, bids in self._bids.items():
                if not bids:
                    continue
                best = min(bids, key=lambda b: b.cost)
                winners.append(Assignment(t, best.robot, best.cost))
                # 勝者通知
                self.comm.broadcast({"type": "assign", "task": t, "robot": best.robot})
            self._bids.clear()
        return winners