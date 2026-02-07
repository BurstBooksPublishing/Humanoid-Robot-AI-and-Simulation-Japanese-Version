import cvxpy as cp
import numpy as np
from typing import List, Optional, Tuple

def compute_safe_joint_update(
    J_task: np.ndarray,
    v_task: np.ndarray,
    Jd_list: List[np.ndarray],
    d_list: List[float],
    q: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    d_safe: float = 0.05,
    lambda_reg: float = 1e-3,
    solver: Optional[str] = None,
    max_iter: int = 4000,
    eps_abs: float = 1e-6,
    eps_rel: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, bool, str]:
    """
    衝突回避と関節限界を考慮した最適関節増分を計算する．
    戻り値: (dq_cmd, success, status)
    """
    m, n = J_task.shape
    assert J_task.ndim == 2 and v_task.ndim == 1 and v_task.size == m
    assert q.size == n and q_min.size == n and q_max.size == n
    assert len(Jd_list) == len(d_list)
    assert np.all(q_min <= q) and np.all(q <= q_max), "初期関節角が限界外"

    dq = cp.Variable(n)

    # タスク追従＋正則化
    objective = cp.Minimize(
        cp.sum_squares(J_task @ dq - v_task) + lambda_reg * cp.sum_squares(dq)
    )

    # 衝突回避：Jd @ dq >= d_safe - d
    collision_constraints = [
        -Jd @ dq <= d - d_safe for Jd, d in zip(Jd_list, d_list)
    ]

    # 関節限界
    joint_limits = [q + dq >= q_min, q + dq <= q_max]

    prob = cp.Problem(objective, collision_constraints + joint_limits)

    # ソルバ選択
    if solver is None:
        solver = cp.OSQP

    try:
        prob.solve(
            solver=solver,
            warm_start=True,
            max_iter=max_iter,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            verbose=verbose,
        )
    except Exception as e:
        return np.zeros(n), False, f"Solver error: {e}"

    if dq.value is None:
        return np.zeros(n), False, prob.status

    return dq.value, True, prob.status