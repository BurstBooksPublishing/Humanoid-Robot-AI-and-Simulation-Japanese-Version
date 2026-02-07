import heapq
import math
from typing import Tuple, Dict, List, Iterable, Optional

Node = Tuple[int, int, int]  # (x, y, theta_index)

class GridMap:
    """軽量インターフェース：占有・クリアランス両方を返す"""
    def __init__(self, occ, clearance):
        self.occ = occ
        self.clearance = clearance
        self.h, self.w = occ.shape

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def is_occupied(self, x: int, y: int) -> bool:
        return bool(self.occ[y, x])

    def clearance_at(self, x: int, y: int) -> float:
        return float(self.clearance[y, x])


def heuristic(n: Node, goal: Node, alpha: float = 1.0, beta: float = 0.5) -> float:
    dx = n[0] - goal[0]
    dy = n[1] - goal[1]
    dist = math.hypot(dx, dy)
    dtheta = abs(((n[2] - goal[2] + math.pi) % (2 * math.pi)) - math.pi)
    return alpha * dist + beta * dtheta


def neighbors(node: Node, grid: GridMap, theta_steps: int = 8) -> Iterable[Node]:
    x, y, t = node
    # 8近傍＋回転3通り
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (1, 1), (-1, 1), (1, -1)]:
        nx, ny = x + dx, y + dy
        if not grid.in_bounds(nx, ny) or grid.is_occupied(nx, ny):
            continue
        for dt in (-1, 0, 1):
            nt = (t + dt) % theta_steps
            yield (nx, ny, nt)


def edge_cost(u: Node, v: Node, clearance_grid: GridMap, max_step: float = 1.0) -> float:
    dist = math.hypot(v[0] - u[0], v[1] - u[1])
    clear_pen = max(0.0, 1.0 - clearance_grid.clearance_at(v[0], v[1]))
    turn = abs(v[2] - u[2])
    return dist + 2.0 * clear_pen + 0.2 * turn


def astar(start: Node, goal: Node, grid: GridMap) -> Optional[List[Node]]:
    open_set: List[Tuple[float, Node]] = []
    heapq.heappush(open_set, (0.0, start))

    g: Dict[Node, float] = {start: 0.0}
    parent: Dict[Node, Node] = {}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            break

        for nbr in neighbors(current, grid):
            tentative = g[current] + edge_cost(current, nbr, grid)
            if tentative < g.get(nbr, math.inf):
                g[nbr] = tentative
                f = tentative + heuristic(nbr, goal)
                heapq.heappush(open_set, (f, nbr))
                parent[nbr] = current
    else:
        return None  # 経路なし

    # 経路再構成
    path = []
    cur = goal
    while cur in parent:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    return path[::-1]