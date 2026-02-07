import numpy as np
from scipy.optimize import least_squares
from typing import Dict, List, Any

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """クォータニオン q でベクトル v を回転"""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    vx, vy, vz = v
    return np.array([
        (1 - 2*y*y - 2*z*z)*vx + 2*(x*y - w*z)*vy + 2*(x*z + w*y)*vz,
        2*(x*y + w*z)*vx + (1 - 2*x*x - 2*z*z)*vy + 2*(y*z - w*x)*vz,
        2*(x*z - w*y)*vx + 2*(y*z + w*x)*vy + (1 - 2*x*x - 2*y*y)*vz
    ])

def logSO3(R: np.ndarray) -> np.ndarray:
    """SO(3) 行列の対数マップ（3次元軸角）"""
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(tr)
    if np.abs(theta) < 1e-6:
        return np.zeros(3)
    return theta / (2*np.sin(theta)) * np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ])

def reprojection_residuals(x: np.ndarray, observations: Dict[str, List[Any]]) -> np.ndarray:
    """
    再投影・IMU 回転残差を計算
    x: [fx, fy, cx, cy, tx, ty, tz, qx, qy, qz, qw]
    """
    K = np.array([[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1]])
    t_cam = x[4:7]
    q_cam = x[7:11]

    res: List[float] = []

    # 画像再投影残差
    for obs in observations.get('images', []):
        X_body = np.asarray(obs['X_body'])
        u_meas = np.asarray(obs['u'])
        X_cam = quat_rotate(q_cam, X_body - t_cam)
        if X_cam[2] <= 0:
            res.extend([1e3, 1e3])
            continue
        u_pred = (K @ (X_cam / X_cam[2]))[:2]
        res.extend(u_meas - u_pred)

    # IMU 回転残差
    for imu_obs in observations.get('imu', []):
        R_pred = np.asarray(imu_obs['R_pred'])
        R_imu  = np.asarray(imu_obs['R_imu'])
        res.extend(logSO3(R_pred.T @ R_imu))

    return np.array(res)

# 初期推定値（例：ID カメラ、零並進、単位クォータニオン）
x0 = np.array([500., 500., 320., 240., 0., 0., 0., 0., 0., 0., 1.])

# 観測データ（外部で用意）
# observations = {'images': [...], 'imu': [...]}

sol = least_squares(
    reprojection_residuals,
    x0,
    args=(observations,),
    method='trf',
    loss='huber',
    verbose=2
)
# sol.x が推定パラメータ