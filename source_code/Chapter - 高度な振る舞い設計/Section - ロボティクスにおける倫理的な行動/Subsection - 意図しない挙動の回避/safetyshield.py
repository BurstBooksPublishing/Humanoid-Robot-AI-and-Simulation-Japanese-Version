import numpy as np
from typing import Callable, Optional

State = np.ndarray
Control = np.ndarray

def safety_filter(
    x: State,
    u_des: Control,
    h: Callable[[State], float],
    dh_dx: Callable[[State], np.ndarray],
    f: Callable[[State], State],
    g: Callable[[State], np.ndarray],
    gamma: float = 5.0,
    tol: float = 1e-6,
    max_corr: Optional[float] = None,
) -> Control:
    """
    1次元安全制約に対する解析的QP投影．
    制約:  dh_dx @ (f + g u) + gamma * h >= 0
    """
    hx = h(x)
    dhx = dh_dx(x)
    fx = f(x)
    gx = g(x)

    a = float(dhx @ fx)
    b = dhx @ gx  # (1, m)

    alpha = gamma * hx
    rhs = -a - alpha
    lhs = float(b @ u_des)

    # すでに安全
    if lhs >= rhs - tol:
        return u_des

    # 半空間への射影
    b_norm_sq = float(b @ b.T) + tol
    scale = (rhs - lhs) / b_norm_sq
    u_corr = u_des + scale * b.T

    # 補正量が許容範囲を超える場合は零指令にフォールバック
    if max_corr is not None:
        du = np.linalg.norm(u_corr - u_des)
        if du > max_corr:
            return np.zeros_like(u_des)

    return u_corr