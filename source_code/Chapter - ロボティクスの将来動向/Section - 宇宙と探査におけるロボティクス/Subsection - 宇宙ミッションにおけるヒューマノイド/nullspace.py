import numpy as np
from typing import Optional, Tuple

def compute_whole_body_torque(
    J: np.ndarray,
    M: np.ndarray,
    q_dot: np.ndarray,
    f_des: np.ndarray,
    tau_posture_des: np.ndarray,
    torque_limits: Optional[Tuple[float, float]] = None,
    damping: float = 1e-4
) -> np.ndarray:
    """
    エンドエフェクタ力とポーズ制御トルクを統合し、関節トルク指令を生成
    """
    if torque_limits is None:
        torque_limits = (-100.0, 100.0)

    # タスク空間トルク: J^T * f_des
    tau_task = J.T @ f_des

    # ダンピング擬似逆行列によるヌル空間射影
    JJt = J @ J.T
    JJt += damping * np.eye(JJt.shape[0])
    J_pinv = np.linalg.solve(JJt, J).T  # (n, 6)
    N = np.eye(J.shape[1]) - J_pinv @ J  # ヌル空間射影行列

    # 最終トルク指令
    tau_cmd = tau_task + N @ tau_posture_des

    # 飽和処理
    tau_cmd = np.clip(tau_cmd, *torque_limits)

    return tau_cmd