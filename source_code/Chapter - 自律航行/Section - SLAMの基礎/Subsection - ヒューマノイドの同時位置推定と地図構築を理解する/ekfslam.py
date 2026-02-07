import numpy as np
from typing import Tuple, List, Iterable, Optional
import scipy.linalg as la

# 状態: [px,py,pz,qx,qy,qz,qw, m1x,m1y,...] ; P 共分散
StateVector = np.ndarray
CovMatrix   = np.ndarray
ImuSample   = np.ndarray
Measurement = Tuple[np.ndarray, int]  # (z, landmark_id)

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数積 (Hamilton product)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_inv(q: np.ndarray) -> np.ndarray:
    """四元数逆"""
    return np.array([q[0], -q[1], -q[2], -q[3]]) / (q @ q)

def integrate_pose(pose: np.ndarray, imu: ImuSample, dt: float) -> np.ndarray:
    """SE(3)上でIMUを積分（簡易版）"""
    p, q = pose[:3], pose[3:7]
    acc, gyro = imu[:3], imu[3:6]
    # 角速度→四元数更新
    dq = np.array([1.0, 0.5*gyro[0]*dt, 0.5*gyro[1]*dt, 0.5*gyro[2]*dt])
    q_new = quat_multiply(q, dq)
    q_new /= la.norm(q_new)
    # 加速度→速度→位置（重力補償省略）
    p_new = p + 0.5 * acc * dt**2
    return np.concatenate([p_new, q_new])

def compute_F(pose: np.ndarray, imu: ImuSample, dt: float) -> np.ndarray:
    """状態遷移ヤコビアン（簡易）"""
    dim = len(pose)
    F = np.eye(dim)
    # 位置・姿勢の線形化（簡略）
    F[:3, :3] += np.eye(3) * 0.01  # ダミー
    return F

def measurement_model(state: StateVector, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """ランドマーク観測モデル（方位＋距離）"""
    p = state[:3]
    lm  = state[7+3*idx:7+3*(idx+1)]
    d   = lm - p
    dist = la.norm(d)
    z_hat = np.array([d[0]/dist, d[1]/dist, d[2]/dist, dist])
    # 観測ヤコビアン（簡易）
    H = np.zeros((4, len(state)))
    H[:3, :3] = -np.eye(3)/dist
    H[:3, 7+3*idx:7+3*(idx+1)] = np.eye(3)/dist
    H[3, :3]  = -d/dist
    H[3, 7+3*idx:7+3*(idx+1)] = d/dist
    return z_hat, H

def predict(state: StateVector, P: CovMatrix, imu: ImuSample,
            dt: float, Q: CovMatrix) -> Tuple[StateVector, CovMatrix]:
    """予測ステップ"""
    pose = state[:7]
    new_pose = integrate_pose(pose, imu, dt)
    F = compute_F(pose, imu, dt)
    state[:7] = new_pose
    P = F @ P @ F.T + Q
    return state, P

def update(state: StateVector, P: CovMatrix, z: np.ndarray, R: CovMatrix,
           landmark_idx: int) -> Tuple[StateVector, CovMatrix]:
    """更新ステップ"""
    h, H = measurement_model(state, landmark_idx)
    y = z - h
    S = H @ P @ H.T + R
    K = la.solve(S, H @ P).T  # 数値安定版
    state += K @ y
    I_KH = np.eye(P.shape[0]) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R @ K.T  # Joseph form
    return state, P

def init_state() -> StateVector:
    """初期状態"""
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

def init_cov(dim: int = 7) -> CovMatrix:
    """初期共分散"""
    return np.eye(dim) * 1.0

def sensor_stream() -> Iterable[Tuple[ImuSample, List[Measurement], float]]:
    """シミュレーション／ROS 2 トピックからIMUと特徴観測を供給"""
    # 実際にはROS 2 subscriber等で置換
    while True:
        yield np.zeros(6), [], 0.01

def main() -> None:
    state = init_state()
    P = init_cov()
    Q_imu = np.eye(7) * 0.01
    R_feat = np.eye(4) * 0.05

    for imu, measurements, dt in sensor_stream():
        state, P = predict(state, P, imu, dt, Q_imu)
        for z, idx in measurements:
            state, P = update(state, P, z, R_feat, idx)

if __name__ == "__main__":
    main()