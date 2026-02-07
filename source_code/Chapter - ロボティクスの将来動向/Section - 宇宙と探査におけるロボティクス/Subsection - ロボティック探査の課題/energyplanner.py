import math
from typing import List, Tuple, Optional, NamedTuple
import logging

# ROS 2 (rclpy) 対応
try:
    import rclpy
    from geometry_msgs.msg import Pose, Twist
except ImportError:
    rclpy = None
    Pose = None
    Twist = None

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotState(NamedTuple):
    """ロボットの状態（位置・姿勢・速度）"""
    pose: Tuple[float, float, float]  # x, y, yaw [rad]
    velocity: Tuple[float, float, float]  # vx, vy, omega [rad/s]


class Waypoint(NamedTuple):
    """ウェイポイント情報"""
    pose: Tuple[float, float, float]
    science_value: float  # 科学的価値（0.0-1.0）


def estimate_overhead(wp: Waypoint) -> float:
    """
    センサ・通信オーバヘッドを推定（単位：ジュール）
    """
    # 簡易モデル：距離に応じた通信コスト + 観測対象の複雑さ
    base_comm = 5.0  # 基本通信コスト
    obs_complexity = 1.0 + wp.science_value  # 科学的価値が高いほど観測コスト増
    return base_comm * obs_complexity


def select_next_waypoint(
    state: RobotState,
    waypoints: List[Tuple[Waypoint, float]],
    battery_joules: float,
    safety_factor: float = 0.15
) -> Optional[Waypoint]:
    """
    バッテリ残量と科学的価値を考慮して次のウェイポイントを選択
    """
    if battery_joules <= 0:
        logger.warning("バッテリ残量が0以下です")
        return None

    safe_margin = safety_factor * battery_joules
    candidates: List[Tuple[Waypoint, float]] = []

    for wp, cost in waypoints:
        overhead = estimate_overhead(wp)
        total_cost = cost + overhead

        # 安全マージンを考慮
        if total_cost + safe_margin <= battery_joules:
            # 科学的価値で重み付け（価値が高いほどコストを下げる）
            weighted_cost = total_cost / max(wp.science_value, 0.01)
            candidates.append((wp, weighted_cost))

    if not candidates:
        logger.info("実行可能なウェイポイントがありません")
        return None

    # 重み付きコスト最小を選択
    best_wp, _ = min(candidates, key=lambda x: x[1])
    logger.info(f"次のウェイポイントを選択: {best_wp.pose}")
    return best_wp