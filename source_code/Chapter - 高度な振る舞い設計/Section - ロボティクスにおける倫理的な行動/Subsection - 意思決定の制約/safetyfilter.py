import numpy as np
import osqp
from scipy import sparse
from typing import List, Tuple, Optional

# 障害物型定義
Obstacle = Tuple[np.ndarray, np.ndarray]  # (mean, cov)

def solve_safety_qp(
    u_nom: np.ndarray,
    x: np.ndarray,
    obst: List[Obstacle],
    v_max: np.ndarray = np.array([0.8, 1.0, 0.5]),
    v_min: np.ndarray = np.array([-0.8, -1.0, -0.5])
) -> np.ndarray:
    """
    速度指令 u_nom を安全な速度 u に修正する QP フィルタ
    """
    nu = len(u_nom)
    P = sparse.csc_matrix(2.0 * np.eye(nu))
    q = -2.0 * u_nom

    # 速度上下限
    A_vel = np.vstack([np.eye(nu), -np.eye(nu)])
    b_vel = np.hstack([v_max, -v_min])

    A_list, b_list = [A_vel], [b_vel]

    # 衝突回避制約の追加
    for mean, cov in obst:
        n, beta = linearize_collision_constraint(x, mean, cov)
        A_list.append(n.reshape(1, -1))
        b_list.append(np.array([beta]))

    A = sparse.csc_matrix(np.vstack(A_list))
    b = np.hstack(b_list)

    # OSQP ソルバ設定
    prob = osqp.OSQP()
    prob.setup(P, q, A, b, verbose=False, polish=True, eps_abs=1e-4, eps_rel=1e-4)
    res = prob.solve()

    # 求解失敗時は公称指令を返す
    return res.x if res.info.status == 'solved' else u_nom


def linearize_collision_constraint(
    x: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    d_min: float = 0.3
) -> Tuple[np.ndarray, float]:
    """
    障害物に対する線形化された衝突回避制約を返す
    """
    # 相対位置と距離
    dx = mean[:2] - x[:2]
    dist = np.linalg.norm(dx)
    if dist < 1e-6:
        dx = np.array([1.0, 0.0])
        dist = 1e-6

    # 法線ベクトル
    n = dx / dist

    # 速度空間での制約係数（回転成分はゼロ）
    J = np.array([n[0], n[1], 0.0])

    # 安全距離を考慮したバイアス
    beta = (d_min - dist) / 0.1  # 0.1[s] 予見時間

    return J, beta