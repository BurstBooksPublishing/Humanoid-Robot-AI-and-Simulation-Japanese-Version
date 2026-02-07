#!/usr/bin/env python3
import random
import math
from typing import Dict, List, Tuple, Optional

# タスク定義: (名前, 楽観a, 最頻m, 悲観b, 通常費用, 短縮費用, 短縮可能週数)
Task = Tuple[str, float, float, float, float, Optional[float], int]

TASKS: List[Task] = [
    ('Frame',        5,  7, 10,  8000, None,  0),
    ('Actuators',    8, 12, 16, 50000, 65000, 0),
    ('PCB',          2,  4,  6,  4000,  7000, 0),
    ('ControlSW',    6, 10, 14, 60000, None,  0),
    ('Integration',  4,  6, 10, 20000, None,  0),
]

# 先行関係 (隣接リスト)
PRECEDENCE: Dict[str, List[str]] = {
    'Frame':       [],
    'Actuators':   ['Frame'],
    'PCB':         ['Frame'],
    'ControlSW':   ['Actuators', 'PCB'],
    'Integration': ['ControlSW'],
}

# タスク名→インデックスの辞書
NAME2IDX: Dict[str, int] = {t[0]: i for i, t in enumerate(TASKS)}

def sample_duration(a: float, m: float, b: float) -> float:
    """PERT近似でタスク期間をサンプリング"""
    mean = (a + 4 * m + b) / 6.0
    sigma = (b - a) / 6.0
    return max(0.1, random.gauss(mean, sigma))

def critical_path_duration(sampled: Dict[str, float]) -> float:
    """動的計画法で最長パス(クリティカルパス)を計算"""
    longest: Dict[str, float] = {}
    for name, _, _, _, _, _, _ in TASKS:
        start = max((longest[p] for p in PRECEDENCE[name]), default=0.0)
        longest[name] = start + sampled[name]
    return max(longest.values())

def monte_carlo(
    deadline_weeks: int = 40,
    n_samples: int = 5000,
    penalty_rate: float = 0.10,
) -> Tuple[float, float]:
    """モンテカルロシミュレーションで納期遵守率と期待コストを推定"""
    base_cost = sum(t[4] for t in TASKS)
    on_time = 0
    total_cost = 0.0

    for _ in range(n_samples):
        sampled = {t[0]: sample_duration(t[1], t[2], t[3]) for t in TASKS}
        dur = critical_path_duration(sampled)
        penalty = base_cost * penalty_rate * max(0.0, (dur - deadline_weeks) / deadline_weeks)
        total_cost += base_cost + penalty
        if dur <= deadline_weeks:
            on_time += 1

    return on_time / n_samples, total_cost / n_samples

if __name__ == '__main__':
    random.seed(42)
    p_on_time, mean_cost = monte_carlo()
    print(f"P(on time)={p_on_time:.3f}  mean cost={mean_cost:,.0f}")