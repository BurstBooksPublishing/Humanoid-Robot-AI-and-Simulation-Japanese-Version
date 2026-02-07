import numpy as np
from typing import Optional, Sequence, Tuple

Array3 = Tuple[float, float, float]


def compute_zmp(
    contact_forces: Sequence[Array3],
    contact_points: Sequence[Array3],
    contact_torques: Sequence[Array3],
    *,
    min_total_force: float = 1e-6,
) -> Optional[Tuple[float, float]]:
    """
    各接触点の力・位置・トルクから水平面ZMPを計算する．

    Parameters
    ----------
    contact_forces  : 各接触点の力 (fx, fy, fz) [N]
    contact_points  : 各接触点の位置 (x, y, z) [m]
    contact_torques : 各接触点のローカルトルク (tx, ty, tz) [N⋅m]
    min_total_force : 分母のfz合計がこれ以下なら接地なしとみなす

    Returns
    -------
    (px, py) : 水平面ZMP位置 [m]．接地力が小さすぎる場合はNone．
    """
    forces = np.asarray(contact_forces, dtype=np.float64)   # (N,3)
    points = np.asarray(contact_points, dtype=np.float64)   # (N,3)
    torques = np.asarray(contact_torques, dtype=np.float64) # (N,3)

    if not (forces.shape == points.shape == torques.shape):
        raise ValueError("forces/points/torques must have the same shape")

    fz = forces[:, 2]
    denom = fz.sum()
    if abs(denom) < min_total_force:
        return None  # 接地なし

    # 各点でのz軸周りモーメント: (r x f)_z + tau_z
    mz = np.cross(points, forces)[:, 2] + torques[:, 2]

    # ZMP座標（水平面）
    px = (points[:, 1] * fz - mz * 0.0).sum() / denom
    py = (-points[:, 0] * fz + mz * 0.0).sum() / denom
    return float(px), float(py)