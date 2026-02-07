from __future__ import annotations
import enum
from typing import Callable, List, Optional, Protocol


class Status(enum.IntEnum):
    """動作の返り値"""
    SUCCESS = 0
    FAILURE = 1
    RUNNING = 2


class Node(Protocol):
    def tick(self) -> Status: ...


ConditionFn = Callable[[], bool]
StartFn   = Callable[[], None]
UpdateFn  = Callable[[], Status]


class Condition(Node):
    """条件判定ノード"""
    def __init__(self, check: ConditionFn) -> None:
        self._check = check

    def tick(self) -> Status:
        return Status.SUCCESS if self._check() else Status.FAILURE


class Action(Node):
    """ROSアクションラッパー"""
    def __init__(self, start: StartFn, update: UpdateFn) -> None:
        self._start   = start
        self._update  = update
        self._started = False

    def tick(self) -> Status:
        if not self._started:
            self._start()
            self._started = True
        return self._update()

    def reset(self) -> None:
        self._started = False


class Sequence(Node):
    """子を順番に実行"""
    def __init__(self, children: List[Node]) -> None:
        self._children = children
        self._index    = 0

    def tick(self) -> Status:
        while self._index < len(self._children):
            status = self._children[self._index].tick()
            if status == Status.RUNNING:
                return Status.RUNNING
            if status == Status.FAILURE:
                self._index = 0
                return Status.FAILURE
            self._index += 1
        self._index = 0
        return Status.SUCCESS


# ------------------------------------------------------------------
# 以下は利用例（実際のROSノードでは別ファイルに配置）
# ------------------------------------------------------------------
def check_visibility() -> bool: ...
def imu_balance_check() -> bool: ...
def plan_trajectory_start() -> None: ...
def plan_trajectory_update() -> Status: ...
def execute_motion_start() -> None: ...
def execute_motion_update() -> Status: ...
def close_gripper_start() -> None: ...
def close_gripper_update() -> Status: ...


def build_pick_tree() -> Node:
    return Sequence([
        Condition(check_visibility),
        Condition(imu_balance_check),
        Action(plan_trajectory_start, plan_trajectory_update),
        Action(execute_motion_start, execute_motion_update),
        Action(close_gripper_start, close_gripper_update),
    ])