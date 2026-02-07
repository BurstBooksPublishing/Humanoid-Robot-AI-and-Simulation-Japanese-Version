from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Protocol, runtime_checkable

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Status(Enum):
    SUCCESS = auto()
    RUNNING = auto()
    FAILURE = auto()


@runtime_checkable
class Node(Protocol):
    def tick(self, blackboard: Dict) -> Status: ...


@dataclass
class Parallel(Node):
    children: List[Node]
    k: int = field(init=False)

    def __post_init__(self):
        n = len(self.children)
        if n == 0:
            raise ValueError("子ノードが空です")
        self.k = n  # デフォルトは全成功

    @classmethod
    def with_ratio(cls, children: List[Node], p: float) -> "Parallel":
        inst = cls(children)
        inst.k = max(1, int(p * len(children)))
        return inst

    def tick(self, blackboard: Dict) -> Status:
        successes = 0
        for child in self.children:
            status = child.tick(blackboard)
            if status == Status.SUCCESS:
                successes += 1
            elif status == Status.RUNNING:
                return Status.RUNNING
        return Status.SUCCESS if successes >= self.k else Status.FAILURE


@dataclass
class Sequence(Node):
    children: List[Node] = field(default_factory=list)

    def tick(self, blackboard: Dict) -> Status:
        for child in self.children:
            status = child.tick(blackboard)
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS


@dataclass
class DynamicManager(Node):
    root: Sequence = field(default_factory=Sequence)

    def inject(self, subtree: Node) -> None:
        self.root.children.append(subtree)
        logger.info("Subtree injected")

    def remove(self, subtree: Node) -> None:
        try:
            self.root.children.remove(subtree)
            logger.info("Subtree removed")
        except ValueError:
            logger.warning("指定されたSubtreeは存在しません")

    def tick(self, blackboard: Dict) -> Status:
        return self.root.tick(blackboard)


@dataclass
class CheckBalance(Node):
    threshold: float = 0.5

    def tick(self, blackboard: Dict) -> Status:
        stability = blackboard.get("stability", 0.0)
        return Status.SUCCESS if stability > self.threshold else Status.FAILURE


@dataclass
class PlanFootstep(Node):
    def tick(self, blackboard: Dict) -> Status:
        # 実際の計画処理は非同期で複数tickを要する
        return Status.RUNNING


def main() -> None:
    blackboard: Dict[str, float] = {"stability": 0.6}
    p = Parallel.with_ratio(
        [CheckBalance(), CheckBalance(), CheckBalance()], p=0.66
    )
    mgr = DynamicManager()
    mgr.inject(p)
    mgr.inject(PlanFootstep())
    print(mgr.tick(blackboard))


if __name__ == "__main__":
    main()