import time
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

# ROS2 関連（必要に応じてアンコメント）
# import rclpy
# from geometry_msgs.msg import PoseStamped
# from nav_msgs.msg import Path

@dataclass
class State:
    x: float
    y: float
    theta: float
    v: float

@dataclass
class MotionPlan:
    traj: List[State]
    footsteps: List[np.ndarray]  # 各足位置のリスト

def plan_to_goal(
    start: State,
    goal: State,
    costmap: np.ndarray,
    dyn_preds: List[np.ndarray],
    max_plan_time: float = 1.0,
    replan_threshold: int = 3
) -> Optional[MotionPlan]:
    """
    グローバル経路をA*で生成し、RRT*で局所軌道を最適化して足位置を決定
    """
    # A* でグローバルコリドーを取得
    corridor = astar_global(start, goal, costmap)
    if not corridor:
        return None

    # RRT* 初期化
    tree = RRTStar(state_dim=4)
    tree.add_root(start)

    start_time = time.time()
    replan_count = 0

    while (time.time() - start_time) < max_plan_time:
        # バイアス付きサンプリング
        x_rand = sample_in_corridor(corridor)
        x_near = tree.nearest(x_rand)
        x_new = steer_kinodynamic(x_near, x_rand)

        # 衝突チェック
        if not collision_check_kin(x_new, costmap):
            continue
        if time_collides_with_preds(x_new, dyn_preds):
            continue

        tree.add_node(x_new, parent=x_near)

        # ゴール到達判定
        if tree.has_solution(goal):
            break

    if not tree.has_solution(goal):
        return None

    traj = tree.reconstruct_path(goal)
    footsteps = footstep_planner(traj)

    # 各足位置を検証
    for fs in footsteps:
        if not ik_solve(fs):
            fs_adj = adjust_footstep(fs)
            if not ik_solve(fs_adj):
                replan_count += 1
                if replan_count >= replan_threshold:
                    return None
                continue
        if not stability_margin_ok(fs):
            return None

    return assemble_motion_plan(traj, footsteps)