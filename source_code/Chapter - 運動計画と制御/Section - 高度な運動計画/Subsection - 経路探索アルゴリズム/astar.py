import heapq
import numpy as np
from typing import Tuple, List, Dict, Optional, Generator, Set

Coord = Tuple[int, int]


def heuristic(a: Coord, b: Coord) -> float:
    """ユークリッド距離"""
    return np.hypot(a[0] - b[0], a[1] - b[1])


def neighbors(node: Coord, grid: np.ndarray) -> Generator[Coord, None, None]:
    """8近傍で占有されていないセルを返す"""
    x, y = node
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (1, 1), (1, -1), (-1, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0:
            yield (nx, ny)


def proximity_cost(node: Coord, dist_map: np.ndarray) -> float:
    """障害物までの距離に応じたペナルティ"""
    d = dist_map[node]
    return max(0.0, 0.5 - d) * 10.0 if d <= 0.5 else 0.0


def astar(start: Coord, goal: Coord, grid: np.ndarray,
          dist_map: np.ndarray) -> List[Coord]:
    """A*で最短経路を返す（経路なしの場合は空リスト）"""
    open_q: List[Tuple[float, float, Coord, Optional[Coord]]] = []
    heapq.heappush(open_q, (heuristic(start, goal), 0.0, start, None))

    came_from: Dict[Coord, Optional[Coord]] = {}
    gscore: Dict[Coord, float] = {start: 0.0}
    closed_set: Set[Coord] = set()

    while open_q:
        _, g, current, parent = heapq.heappop(open_q)
        if current in closed_set:
            continue
        closed_set.add(current)
        came_from[current] = parent

        if current == goal:
            break

        for nbr in neighbors(current, grid):
            step = heuristic(current, nbr)
            tentative_g = g + step + proximity_cost(nbr, dist_map)
            if tentative_g < gscore.get(nbr, np.inf):
                gscore[nbr] = tentative_g
                heapq.heappush(open_q,
                               (tentative_g + heuristic(nbr, goal), tentative_g, nbr, current))

    # 経路再構築
    path: List[Coord] = []
    n: Optional[Coord] = goal
    while n is not None:
        path.append(n)
        n = came_from.get(n)
    return path[::-1] if path[-1] == start else []