import cvxpy as cp
import numpy as np
from typing import Optional, Tuple

class SingleStepMPC:
    """
    1ステップ最適トルク指令を計算する軽量MPC
    """
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        tau_min: np.ndarray,
        tau_max: np.ndarray,
        solver: str = cp.OSQP,
    ) -> None:
        assert A.shape == (6, 6)
        assert B.shape == (6, 12)
        assert Q.shape == (6, 6)
        assert R.shape == (12, 12)
        assert tau_min.shape == (12,)
        assert tau_max.shape == (12,)

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.solver = solver

        # 決定変数とパラメータを初期化
        self.tau = cp.Variable(12)
        self.x = cp.Parameter(6)
        self.x_ref = cp.Parameter(6)

        # コストと制約を構築
        x_next = self.A @ self.x + self.B @ self.tau
        cost = cp.quad_form(self.tau, self.R) + cp.quad_form(x_next - self.x_ref, self.Q)
        constraints = [self.tau_min <= self.tau, self.tau <= self.tau_max]

        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, x: np.ndarray, x_ref: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """1ステップ最適化を実行し、成功フラグと最適トルクを返す"""
        self.x.value = x
        self.x_ref.value = x_ref
        try:
            self.prob.solve(solver=self.solver, warm_start=True)
            if self.prob.status != cp.OPTIMAL:
                return False, None
            return True, self.tau.value
        except Exception:
            return False, None


# 使用例（本ファイルが直接実行された場合）
if __name__ == "__main__":
    A = np.eye(6)
    B = np.zeros((6, 12))
    Q = np.eye(6)
    R = 1e-3 * np.eye(12)
    tau_min = -50.0 * np.ones(12)
    tau_max = 50.0 * np.ones(12)

    mpc = SingleStepMPC(A, B, Q, R, tau_min, tau_max)

    x = np.zeros(6)
    x_ref = np.zeros(6)

    ok, tau_opt = mpc.solve(x, x_ref)
    if ok:
        print("最適トルク指令:", tau_opt)
    else:
        print("最適化失敗")