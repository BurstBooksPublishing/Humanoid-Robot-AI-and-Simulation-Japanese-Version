import py_trees as pt
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from typing import Dict, Any, Callable, Set, Optional
import numpy as np

class TaskAllocator(pt.behaviour.Behaviour):
    """
    タスクの優先度とバッテリ残量を考慮し、実行タスク集合を決定する。
    ROS 2 パラメータサーバ経由で割当結果を公開。
    """
    def __init__(
        self,
        node: Node,
        tasks: Dict[str, Dict[str, Any]],
        battery_provider: Callable[[], float],
        param_ns: str = "task_allocator"
    ):
        super().__init__(name="TaskAllocator")
        self.node = node
        self.tasks = tasks
        self.battery = battery_provider
        self.active: Set[str] = set()
        self._param_ns = param_ns
        self._declare_params()

    # パラメータ宣言
    def _declare_params(self):
        self.node.declare_parameter(f"{self._param_ns}.emergency_preempt", True)
        self.node.declare_parameter(f"{self._param_ns}.safety_check", True)

    # パラメータ取得
    def _get_params(self):
        return {
            "emergency_preempt": self.node.get_parameter(f"{self._param_ns}.emergency_preempt").value,
            "safety_check": self.node.get_parameter(f"{self._param_ns}.safety_check").value,
        }

    def update(self) -> pt.common.Status:
        B = self.battery()
        params = self._get_params()

        # 緊急タスクが存在すれば即座に選択
        if params["emergency_preempt"]:
            for tid, meta in self.tasks.items():
                if meta.get("emergency", False):
                    self.active = {tid}
                    self._publish_allocation()
                    return pt.common.Status.SUCCESS

        # 効用/コスト比で降順ソート
        ordered = sorted(
            self.tasks.items(),
            key=lambda kv: kv[1]["u"] / max(kv[1]["c"], 1e-6),
            reverse=True
        )

        chosen: Set[str] = set()
        used_cost = 0.0
        used_res: Set[str] = set()

        for tid, meta in ordered:
            if used_cost + meta["c"] > B:
                continue
            if used_res & set(meta.get("r", [])):
                continue
            if params["safety_check"] and not meta.get("safety_ok", True):
                continue
            chosen.add(tid)
            used_cost += meta["c"]
            used_res |= set(meta.get("r", []))

        self.active = chosen
        self._publish_allocation()
        return pt.common.Status.SUCCESS

    # 割当結果をROS 2 パラメータとして公開
    def _publish_allocation(self):
        self.node.set_parameters([
            rclpy.parameter.Parameter(
                f"{self._param_ns}.active_tasks",
                rclpy.Parameter.Type.STRING_ARRAY,
                list(self.active)
            )
        ])


class ActivationDecorator(pt.decorators.Decorator):
    """
    allocator.active に含まれる場合のみ子を実行。
    含まれなくなった瞬間に子を中断（cancel）する。
    """
    def __init__(
        self,
        task_id: str,
        allocator: TaskAllocator,
        child: pt.behaviour.Behaviour,
        name: Optional[str] = None
    ):
        name = name or f"Active_{task_id}"
        super().__init__(name=name, child=child)
        self.task_id = task_id
        self.allocator = allocator
        self._was_active = False

    def update(self) -> pt.common.Status:
        active = self.task_id in self.allocator.active
        if active:
            self._was_active = True
            return self.decorated.update()
        else:
            if self._was_active:
                # 一度実行していたら中断
                self.decorated.stop(pt.common.Status.INVALID)
                self._was_active = False
            return pt.common.Status.FAILURE


# ノード初期化 & ツリー構築例
def build_tree(node: Node, tasks_dict: Dict[str, Any], battery_provider: Callable[[], float]) -> pt.behaviour.Behaviour:
    allocator = TaskAllocator(node, tasks_dict, battery_provider)

    supervisor = build_supervisor_subtree(node)

    task_pool = pt.composites.Parallel(
        "TaskPool",
        policy=pt.common.ParallelPolicy.SuccessOnAll(False, False)
    )

    for tid, meta in tasks_dict.items():
        leaf = build_task_subtree(node, tid, meta)
        deco = ActivationDecorator(tid, allocator, leaf)
        task_pool.add_child(deco)

    root = pt.composites.Parallel(
        "Root",
        policy=pt.common.ParallelPolicy.SuccessOnAll(False, False)
    )
    root.add_children([allocator, supervisor, task_pool])
    return root