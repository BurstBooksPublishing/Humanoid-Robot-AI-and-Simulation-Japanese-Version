import math
from typing import Final

# 物理定数
_G: Final[float] = 9.80665  # 重力加速度 [m/s^2]

def safe_velocity(
    distance: float,
    mu_r: float,
    sigma_r: float,
    a_max: float,
    d_human_motion: float = 0.0,
    k: float = 2.0,
) -> float:
    """
    人間との衝突回避を保証する最大速度を返す。
    単位は全て SI 系（m, s, m/s, m/s^2）。
    """
    # 負値ガード
    if distance < 0.0 or mu_r < 0.0 or sigma_r < 0.0 or a_max <= 0.0:
        return 0.0

    margin: float = max(0.0, distance - d_human_motion)
    if margin <= 0.0:
        return 0.0

    tau: float = mu_r + k * sigma_r
    A: float = 1.0 / (2.0 * a_max)
    B: float = tau
    C: float = -margin

    discriminant: float = B * B - 4.0 * A * C
    if discriminant < 0.0:
        return 0.0

    v_max: float = (-B + math.sqrt(discriminant)) / (2.0 * A)
    return max(0.0, v_max)