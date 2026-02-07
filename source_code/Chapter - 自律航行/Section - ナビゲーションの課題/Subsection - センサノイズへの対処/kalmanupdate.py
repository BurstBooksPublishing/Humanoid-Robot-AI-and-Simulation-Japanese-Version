import numpy as np
from typing import Tuple, Optional

def kalman_measurement_update(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    gate_thresh: float = 9.21,
    adaptive: bool = True,
    eps: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    カルマン観測更新（外れ値ゲート付き）
    """
    z_pred = H @ x
    y = z - z_pred
    S = H @ P @ H.T + R

    # 数値安定なコレスキー分解で逆行列を回避
    try:
        L = np.linalg.cholesky(S)
        maha = float(y.T @ np.linalg.solve(S, y))
    except np.linalg.LinAlgError:
        # 正定でなければ疑似逆行列でフォールバック
        S_inv = np.linalg.pinv(S, rcond=eps)
        maha = float(y.T @ S_inv @ y)

    # 外れ値ゲート
    if maha > gate_thresh and adaptive:
        scale = maha / gate_thresh
        R *= scale ** 2  # 共分散を大きくして信頼度を下げる
        S = H @ P @ H.T + R
        L = np.linalg.cholesky(S)

    # カルマンゲイン（コレスキー分解で効率化）
    K = np.linalg.solve(L, np.linalg.solve(L.T, H @ P.T)).T
    x += K @ y
    P = (np.eye(P.shape[0]) - K @ H) @ P

    return x, P, maha