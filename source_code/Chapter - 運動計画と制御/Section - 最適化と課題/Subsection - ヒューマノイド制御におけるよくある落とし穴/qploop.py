import numpy as np
import osqp
from typing import Optional, Tuple

class QPController:
    """MPCレベルで関節トルクをQP最適化するプロダクションコントローラ"""
    def __init__(self,
                 n_j: int,
                 tau_max: np.ndarray,
                 solver_eps_abs: float = 1e-4,
                 solver_eps_rel: float = 1e-4,
                 max_iter: int = 4000):
        self.n_j = n_j
        self.tau_max = tau_max
        self.solver = osqp.OSQP()
        self.solver.settings.update(eps_abs=solver_eps_abs,
                                    eps_rel=solver_eps_rel,
                                    max_iter=max_iter,
                                    warm_start=True,
                                    verbose=False)
        self._last_valid_tau = np.zeros(n_j)

    def update(self,
               cost_hess: np.ndarray,
               cost_grad: np.ndarray,
               A_eq: Optional[np.ndarray],
               b_eq: Optional[np.ndarray],
               C_ineq: Optional[np.ndarray],
               d_ineq: Optional[np.ndarray]) -> np.ndarray:
        """QPを組み立て・解き、安全なトルクを返す"""
        # 正定化
        H = cost_hess + 1e-6 * np.eye(self.n_j)

        # 等式・不等式を結合
        if A_eq is not None and b_eq is not None:
            A = np.vstack([A_eq, C_ineq]) if C_ineq is not None else A_eq
            l = np.hstack([b_eq, -np.inf * np.ones(d_ineq.shape)]) if C_ineq is not None else b_eq
            u = np.hstack([b_eq, d_ineq]) if C_ineq is not None else b_eq
        else:
            A = C_ineq
            l = -np.inf * np.ones(d_ineq.shape)
            u = d_ineq

        # OSQPセットアップ（行列疎化済みと仮定）
        self.solver.setup(P=H, q=cost_grad, A=A, l=l, u=u, warm_start=True)
        res = self.solver.solve()

        if res.info.status_val != self.solver.constant('OSQP_SOLVED'):
            tau_cmd = self._last_valid_tau.copy()
        else:
            tau_cmd = res.x[:self.n_j]
            self._last_valid_tau = tau_cmd.copy()

        # 遅延補償（簡易版）
        tau_cmd = self._latency_compensate(tau_cmd)

        # 安全クリップ
        return np.clip(tau_cmd, -self.tau_max, self.tau_max)

    def _latency_compensate(self, tau: np.ndarray) -> np.ndarray:
        # 実機では測定遅延分の状態予測を行う
        return tau


# 使用例
controller = QPController(n_j=7, tau_max=np.array([100.0]*7))

# 各時刻で呼ぶ
tau_safe = controller.update(build_Hessian(cost_terms),
                               build_gradient(task_errors),
                               *build_equality_constraints(),
                               *build_inequality_constraints())
send_to_actuators(tau_safe)