import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

# グローバルロガー
logger = logging.getLogger(__name__)


def score_candidates(
    height_map: np.ndarray,
    cov_map: np.ndarray,
    robot_model,
    params: Dict[str, float],
    map_index_to_world: callable,
    estimate_normal: callable,
    local_roughness: callable,
) -> List[Dict[str, Tuple[float, float, float] | float]]:
    """
    高さマップと共分散マップから次の脚位置候補をコストでソートして返す。
    """
    if height_map.shape != cov_map.shape[:2]:
        raise ValueError("height_map と cov_map の形状が不一致")

    candidates: List[Dict] = []

    # マルチスレッド化を見越してループはシンプルに保つ
    for (i, j), h in np.ndenumerate(height_map):
        try:
            x, y = map_index_to_world(i, j)
            n = estimate_normal(height_map, i, j)
            theta = np.arccos(np.clip(np.dot(n, [0, 0, 1]), -1.0, 1.0))
            r = local_roughness(height_map, i, j)

            reachable, d_ik = robot_model.ik_reachability(x, y, h)
            if not reachable:
                continue

            zmp_pen = predict_zmp_violation(robot_model, x, y, h)
            uncert = float(np.trace(cov_map[i, j]))

            cost = (
                params["ws"] * theta
                + params["wr"] * r
                + params["wk"] * d_ik
                + params["wz"] * max(0.0, zmp_pen)
                + params["wu"] * uncert
            )

            candidates.append({"pos": (x, y, h), "cost": cost})
        except Exception as e:
            # セーフティ：予期せぬエラーで全体を止めない
            logger.debug(f"skip cell ({i},{j}): {e}")
            continue

    # コスト昇順でソート
    return sorted(candidates, key=lambda c: c["cost"])