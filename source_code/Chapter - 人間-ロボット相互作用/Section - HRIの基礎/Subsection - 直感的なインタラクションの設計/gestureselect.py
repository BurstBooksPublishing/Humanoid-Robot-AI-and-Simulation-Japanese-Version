import numpy as np
from typing import List, Dict, Callable, Any, Tuple

GestureTemplate = Dict[str, Any]
GoalPrior       = Dict[str, float]
GoalLikelihood  = Callable[[np.ndarray], float]

def score_gestures(
    gesture_templates: List[GestureTemplate],
    partial_obs: np.ndarray,
    user_prior: GoalPrior,
    goal_models: Dict[str, GoalLikelihood],
    w_leg: float = 1.0,
    w_safe: float = 1.2,
    w_cost: float = 0.8,
    temperature: float = 2.0,
    rng: np.random.Generator = np.random.default_rng(),
) -> GestureTemplate:
    """
    部分観測に対して最も効用の高い身振りテンプレートを返す
    """
    scores: List[Tuple[float, GestureTemplate]] = []

    for g in gesture_templates:
        traj = g["traj"]
        # 各ゴールに対する事後確率に比例した可読性を集計
        legibility = sum(
            goal_models[goal](partial_obs) * prior
            for goal, prior in user_prior.items()
        )

        # モダリティコストルックアップ
        modality_cost = {"speech": 1.5, "gesture": 1.0, "display": 0.8}.get(
            g["modality"], 1.0
        )
        safety_score = float(np.clip(g.get("safety", 0.0), 0.0, 1.0))

        # 効用関数
        utility = w_leg * legibility + w_safe * safety_score - w_cost * modality_cost
        scores.append((utility, g))

    # ソフトマックスで選択
    utilities = np.array([u for u, _ in scores])
    logits = temperature * (utilities - utilities.max())
    probs = np.exp(logits)
    probs /= probs.sum()

    idx = rng.choice(len(scores), p=probs)
    return scores[idx][1]