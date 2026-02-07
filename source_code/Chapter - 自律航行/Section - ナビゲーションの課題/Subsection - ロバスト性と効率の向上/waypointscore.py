import math
from typing import List, Tuple

import numpy as np
import numpy.typing as npt


class WaypointEvaluator:
    """ウェイポイントを安全性・エネルギー・進捗で評価"""

    def __init__(
        self,
        robot_radius: float = 0.35,
        pos_sigma: float = 0.15,
        lambda_s: float = 5.0,
        lambda_e: float = 1.0,
        lambda_d: float = 0.5,
        energy_factor: float = 10.0,
    ) -> None:
        self.r = robot_radius
        self.sigma = pos_sigma
        self.lambdas = np.array([lambda_s, lambda_e, lambda_d])
        self.energy_factor = energy_factor

    # 衝突確率を計算
    def collision_prob(self, d_min: float) -> float:
        z = (d_min - self.r) / self.sigma
        return 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # エネルギー代用値
    def energy_estimate(self, curr: npt.NDArray, goal: npt.NDArray) -> float:
        return float(np.linalg.norm(goal - curr)) * self.energy_factor

    # スコアを算出
    def score(
        self,
        curr: npt.NDArray,
        wp: npt.NDArray,
        d_min: float,
    ) -> float:
        p_coll = self.collision_prob(d_min)
        e_cost = self.energy_estimate(curr, wp)
        dist = float(np.linalg.norm(wp - curr))
        terms = np.array([p_coll, e_cost, dist])
        return float(np.dot(self.lambdas, terms))

    # 最良ウェイポイントを選択
    def select_best(
        self,
        curr: npt.NDArray,
        candidates: List[npt.NDArray],
        d_mins: List[float],
    ) -> Tuple[npt.NDArray, float]:
        scores = [self.score(curr, wp, dm) for wp, dm in zip(candidates, d_mins)]
        idx = int(np.argmin(scores))
        return candidates[idx], scores[idx]


# 使用例
if __name__ == "__main__":
    evaluator = WaypointEvaluator()

    curr = np.array([0.0, 0.0])
    candidates = [np.array([1.0, 0.0]), np.array([0.8, 0.3]), np.array([0.5, 0.9])]
    d_mins = [0.6, 0.4, 0.8]

    best_wp, best_score = evaluator.select_best(curr, candidates, d_mins)