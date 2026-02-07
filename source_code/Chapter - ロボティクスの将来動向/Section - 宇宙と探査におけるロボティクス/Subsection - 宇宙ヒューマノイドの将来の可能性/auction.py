#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ログ設定：本番環境ではINFO以上、開発時はDEBUGに切り替え
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Agent:
    """エージェントの状態を保持"""
    aid: int
    energy_wh: float
    reserved: bool = field(default=False, init=False)

    def can_perform(self, cost: float) -> bool:
        return not self.reserved and self.energy_wh >= cost


@dataclass(slots=True)
class Task:
    """タスクの情報を保持"""
    tid: str
    utility: float


@dataclass(slots=True)
class Bid:
    """入札情報"""
    agent: Agent
    task: Task
    value: float
    cost: float


class Auctioneer:
    """単純な貪欲オークション実装"""

    def __init__(self, agents: List[Agent], tasks: List[Task]) -> None:
        self.agents = {a.aid: a for a in agents}
        self.tasks = {t.tid: t for t in tasks}

    def estimate_cost(self, agent: Agent, task: Task) -> float:
        """タスク実行に必要な推定エネルギー消費[Wh]"""
        # 実機では経路・動作履歴から推定
        if task.tid == "A":
            return 20.0
        if task.tid == "B":
            return 10.0
        return 15.0  # デフォルト

    def compute_bid_value(self, agent: Agent, task: Task, cost: float) -> float:
        """効用から正規化コストを差し引いた入札値"""
        return task.utility - (cost / agent.energy_wh) * task.utility

    def run(self) -> Dict[str, int]:
        """降順貪欲割当てを実行"""
        bids: List[Bid] = []

        # 全組み合わせで入札を生成
        for agent in self.agents.values():
            for task in self.tasks.values():
                cost = self.estimate_cost(agent, task)
                if agent.can_perform(cost):
                    value = self.compute_bid_value(agent, task, cost)
                    bids.append(Bid(agent, task, value, cost))

        # 入札値の降順でソート
        bids.sort(key=lambda b: b.value, reverse=True)

        allocation: Dict[str, int] = {}
        for bid in bids:
            if bid.agent.reserved or bid.task.tid in allocation:
                continue
            allocation[bid.task.tid] = bid.agent.aid
            bid.agent.reserved = True
            logger.info("Task %s -> Agent %d (value=%.2f)", bid.task.tid, bid.agent.aid, bid.value)

        return allocation


# ---------------- 使用例 ----------------
if __name__ == "__main__":
    agents = [Agent(aid=0, energy_wh=120.0), Agent(aid=1, energy_wh=80.0)]
    tasks  = [Task(tid="A", utility=100.0), Task(tid="B", utility=60.0)]

    auction = Auctioneer(agents, tasks)
    result = auction.run()
    print("Allocation:", result)