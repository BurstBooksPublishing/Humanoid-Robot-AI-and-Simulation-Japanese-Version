import math
import random
from typing import Dict, List, Optional, Tuple

# 3-DoF 姿勢型エイリアス
Pose = Tuple[float, float, float]


class Node:
    """RRT 木のノード：姿勢と親ノードを保持"""
    def __init__(self, q: Pose, parent: Optional["Node"] = None) -> None:
        self.q: Pose = q
        self.parent: Optional["Node"] = parent


def distance(a: Pose, b: Pose) -> float:
    """SE(2) 距離：位置＋角度を重み付けて評価"""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dtheta = math.atan2(math.sin(a[2] - b[2]), math.cos(a[2] - b[2]))
    return math.hypot(dx, dy) + 0.1 * abs(dtheta)


def sample_random(bounds: Dict[str, float]) -> Pose:
    """設定空間から一様ランダムに姿勢をサンプリング"""
    x = random.uniform(bounds["xmin"], bounds["xmax"])
    y = random.uniform(bounds["ymin"], bounds["ymax"])
    theta = random.uniform(-math.pi, math.pi)
    return (x, y, theta)


def nearest(tree: List[Node], q_rand: Pose) -> Node:
    """tree 中で q_rand に最も近いノードを返す"""
    return min(tree, key=lambda n: distance(n.q, q_rand))


def steer(q_from: Pose, q_to: Pose, step: float = 0.2) -> Pose:
    """固定ステップ長で SE(2) 線形補間"""
    dist = distance(q_from, q_to)
    if dist <= step:
        return q_to
    alpha = step / dist
    x = q_from[0] + alpha * (q_to[0] - q_from[0])
    y = q_from[1] + alpha * (q_to[1] - q_from[1])
    theta = q_from[2] + alpha * math.atan2(
        math.sin(q_to[2] - q_from[2]), math.cos(q_to[2] - q_from[2])
    )
    theta = math.atan2(math.sin(theta), math.cos(theta))  # [-pi,pi] 正規化
    return (x, y, theta)


def reconstruct_path(leaf: Node) -> List[Pose]:
    """ゴールノードから根まで逆順にパスを再構成"""
    path: List[Pose] = []
    curr: Optional[Node] = leaf
    while curr is not None:
        path.append(curr.q)
        curr = curr.parent
    path.reverse()
    return path


def plan_rrt(
    q_start: Pose,
    q_goal: Pose,
    bounds: Dict[str, float],
    max_iter: int = 5000,
    goal_sample_rate: float = 0.05,
    step_size: float = 0.2,
    goal_tol: float = 0.3,
) -> Optional[List[Pose]]:
    """
    単純 RRT プランナ
    collision_check は呼び出し側で実装すること
    """
    tree: List[Node] = [Node(q_start)]

    for _ in range(max_iter):
        # ゴールバイアス：確率的に q_goal をサンプリング
        if random.random() < goal_sample_rate:
            q_rand = q_goal
        else:
            q_rand = sample_random(bounds)

        q_near = nearest(tree, q_rand)
        q_new = steer(q_near.q, q_rand, step_size)

        # 衝突チェック（未実装：呼び出し側で定義）
        if not collision_check(q_near.q, q_new):
            continue

        tree.append(Node(q_new, q_near))

        # ゴール到達判定
        if distance(q_new, q_goal) < goal_tol:
            if collision_check(q_new, q_goal):
                return reconstruct_path(Node(q_goal, tree[-1]))

    return None


# 呼び出し側で実装必須
def collision_check(q_from: Pose, q_to: Pose) -> bool:
    """ダミー衝突チェック：常に False（衝突あり）を返す"""
    return False