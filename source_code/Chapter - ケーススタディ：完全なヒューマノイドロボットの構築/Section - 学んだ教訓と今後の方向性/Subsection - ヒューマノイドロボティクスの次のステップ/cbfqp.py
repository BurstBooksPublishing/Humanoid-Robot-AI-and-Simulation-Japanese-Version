import cvxpy as cp
import numpy as np
from typing import Optional

class CBF_QP:
    """
    1次元平面ダイナミクス用のCBF-QP安全フィルタ
    """
    def __init__(self,
                 A: np.ndarray,
                 B: np.ndarray,
                 x_safe_min: float = 0.0,
                 k: float = 5.0,
                 u_max: float = 2.0,
                 solver: Optional[str] = None) -> None:
        self.A = A
        self.B = B
        self.x_safe_min = x_safe_min
        self.k = k
        self.u_max = u_max
        self.solver = solver or cp.OSQP

    def filter(self, x: np.ndarray, u_nom: np.ndarray) -> Optional[np.ndarray]:
        h = x[0] - self.x_safe_min
        Lfh = self.A[0, :].dot(x)
        Lgh = float(self.B[0, :])  # 1x1行列をスカラー化

        u = cp.Variable(1)
        # CBF条件：h_dot + k*h >= 0
        constraints = [Lfh + Lgh * u[0] + self.k * h >= 0,
                       cp.abs(u) <= self.u_max]

        prob = cp.Problem(cp.Minimize(cp.sum_squares(u - u_nom)), constraints)
        try:
            prob.solve(solver=self.solver)
            return u.value if prob.status == cp.OPTIMAL else None
        except Exception:
            return None


# 使用例
if __name__ == "__main__":
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    cbf = CBF_QP(A, B, x_safe_min=0.0, k=5.0, u_max=2.0)

    x = np.array([0.2, -0.1])
    u_nom = np.array([0.5])
    u_safe = cbf.filter(x, u_nom)
    if u_safe is not None:
        print("u_safe =", u_safe[0])