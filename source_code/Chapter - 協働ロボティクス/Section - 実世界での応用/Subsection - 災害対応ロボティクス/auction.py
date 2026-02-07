import numpy as np
from typing import Tuple, Sequence

# センサモデル定数
_SENSOR_GAMMA: float = 0.5          # 観測後のエントロピー減衰率
_NOMINAL_SPEED: float = 0.5         # [m/s]
_EPS: float = 1e-9                  # 対数発散防止クリップ値


def info_gain(grid_prob: np.ndarray, region_idx: Tuple[int, int]) -> float:
    """
    指定領域の観測による情報利得をシャノンエントロピー差分で算出
    """
    p = grid_prob[region_idx]
    # エントロピー計算（0 log 0 を回避）
    H_before = -np.sum(p * np.log(np.clip(p, _EPS, 1.0)))
    H_after = H_before * (1.0 - _SENSOR_GAMMA)
    return H_before - H_after


def energy_cost(robot_pose: Sequence[float], region_centroid: Sequence[float]) -> float:
    """
    ロボット位置から領域重心までの移動時間見積もり
    """
    d = np.linalg.norm(
        np.array(robot_pose[:2], dtype=float) -
        np.array(region_centroid[:2], dtype=float)
    )
    return d / _NOMINAL_SPEED


def estimate_risk(region_centroid: Sequence[float]) -> float:
    """
    ドメイン固有のリスク推定関数（ダミー実装）
    """
    # TODO: 実環境に応じて置換
    return 0.0


def compute_bid(
    robot_pose: Sequence[float],
    grid_prob: np.ndarray,
    region_idx: Tuple[int, int],
    region_centroid: Sequence[float],
    alpha: float = 1.0,
    beta: float = 0.7,
    gamma: float = 1.2
) -> float:
    """
    単一ロボットが領域に対する效用（bid）を算出
    """
    I = info_gain(grid_prob, region_idx)
    E = energy_cost(robot_pose, region_centroid)
    R = estimate_risk(region_centroid)
    return alpha * I - beta * E - gamma * R