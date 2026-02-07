import numpy as np
from typing import Dict, Tuple

# 物理定数
g: float = 9.81
z_com: float = 0.85
omega0: float = np.sqrt(g / z_com)
alpha: float = 0.8

# フィルタ状態
_zmp_filtered: np.ndarray = np.zeros(2)


def compute_zmp(
    ft_left: Dict[str, np.ndarray],
    ft_right: Dict[str, np.ndarray],
    imu_accel: np.ndarray,
    foot_positions: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ZMP 推定とキャプチャポイント予測
    """
    # 合力・合力モーメント
    F: np.ndarray = ft_left["force"] + ft_right["force"]
    M: np.ndarray = ft_left["moment"] + ft_right["moment"]
    Fz: float = F[2] + 1e-6  # ゼロ除算回避

    # ZMP 計算
    zmp_meas: np.ndarray = np.array([-M[1] / Fz, M[0] / Fz])

    # ローパスフィルタ
    global _zmp_filtered
    _zmp_filtered = alpha * _zmp_filtered + (1 - alpha) * zmp_meas

    # CoM 状態（外部推定器から取得想定）
    com_pos: np.ndarray = np.zeros(2)
    com_vel: np.ndarray = np.zeros(2)

    # キャプチャポイント
    cp: np.ndarray = com_pos + com_vel / omega0

    return _zmp_filtered, cp