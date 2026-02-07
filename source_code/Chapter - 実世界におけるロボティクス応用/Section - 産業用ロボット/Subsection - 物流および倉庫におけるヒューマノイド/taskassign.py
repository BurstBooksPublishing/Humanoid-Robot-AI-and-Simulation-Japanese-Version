#!/usr/bin/env python3
import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rmf_fleet_msgs.msg import RobotState, TaskSummary, FleetState
from rmf_task_msgs.msg import TaskProfile, TaskType

# 単一タスクを表す不変オブジェクト
@dataclass(frozen=True)
class Task:
    id: str
    travel: float
    pick: float
    place: float
    priority: int = 0  # 大きいほど優先

# ロボットの内部状態
@dataclass
class Robot:
    id: str
    eta: float = 0.0
    tasks: List[Task] = field(default_factory=list)
    battery: float = 1.0  # 0-1
    needs_swap: bool = False

# タスク予約管理ノード
class FleetManagerNode(Node):
    def __init__(self):
        super().__init__('fleet_manager')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        self.task_sub = self.create_subscription(
            TaskProfile, 'task_requests', self._on_task, qos
        )
        self.fleet_sub = self.create_subscription(
            FleetState, 'fleet_states', self._on_fleet, qos
        )
        self.assign_pub = self.create_publisher(
            TaskSummary, 'task_assignments', qos
        )

        self._robots: Dict[str, Robot] = {}
        self._task_queue: List[Tuple[int, float, Task]] = []  # (-priority, stamp, task)
        self._lock = threading.Lock()
        self._timer = self.create_timer(0.5, self._assign)

    # タスク到着時に優先度付きキューへ追加
    def _on_task(self, msg: TaskProfile):
        task = Task(
            id=msg.task_id,
            travel=msg.description.travel_time,
            pick=msg.description.pick_time,
            place=msg.description.place_time,
            priority=msg.priority
        )
        with self._lock:
            heapq.heappush(self._task_queue, (-task.priority, time.time(), task))

    # FleetState からロボット状態を更新
    def _on_fleet(self, msg: FleetState):
        with self._lock:
            for r in msg.robots:
                if r.name not in self._robots:
                    self._robots[r.name] = Robot(id=r.name)
                self._robots[r.name].battery = r.battery_percent / 100.0
                self._robots[r.name].needs_swap = r.battery_percent < 15.0

    # ETA推定：バッテリ交換時間を考慮
    def _estimate(self, robot: Robot, task: Task) -> float:
        base = robot.eta + task.travel + task.pick + task.place
        if robot.needs_swap:
            base += 300.0  # 5分交換
        return base

    # 割当て実行
    def _assign(self):
        with self._lock:
            while self._task_queue and self._robots:
                _, _, task = heapq.heappop(self._task_queue)
                best = min(self._robots.values(),
                           key=lambda r: self._estimate(r, task))
                best.eta = self._estimate(best, task)
                best.tasks.append(task)

                # 割当て通知
                assign = TaskSummary()
                assign.task_id = task.id
                assign.robot_name = best.id
                self.assign_pub.publish(assign)

def main():
    rclpy.init()
    node = FleetManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()