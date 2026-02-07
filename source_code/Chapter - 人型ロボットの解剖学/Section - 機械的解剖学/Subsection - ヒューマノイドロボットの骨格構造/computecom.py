import numpy as np
from typing import Tuple

def compute_global_com(masses: np.ndarray,
                       com_positions: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    全質量と重心を返す
    """
    if masses.ndim != 1 or com_positions.ndim != 2:
        raise ValueError("massesは1次元、com_positionsは2次元で与える")
    if masses.shape[0] != com_positions.shape[0]:
        raise ValueError("massesとcom_positionsの長さが一致しない")
    if masses.min() <= 0:
        raise ValueError("質量は正の値")

    M = masses.sum()
    global_com = (masses[:, None] * com_positions).sum(axis=0) / M
    return float(M), global_com


if __name__ == "__main__":
    masses = np.array([12.0, 3.5, 2.0, 2.0, 4.0, 1.2, 1.2])
    com_positions = np.array([
        [0.0, 0.0, 0.45],
        [0.0, 0.12, 0.0],
        [0.0, -0.12, 0.0],
        [0.0, 0.12, -0.4],
        [0.0, -0.12, -0.4],
        [0.2, 0.15, 0.1],
        [0.2, -0.15, 0.1],
    ])

    M, global_com = compute_global_com(masses, com_positions)
    print("Total mass:", M)
    print("Global COM (m):", global_com)