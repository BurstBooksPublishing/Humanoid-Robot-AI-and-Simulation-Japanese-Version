import numpy as np
from typing import Union

Number = Union[int, float]

def estimate_energy(
    torque: np.ndarray,
    omega: np.ndarray,
    dt: float,
    R: Number,
    k_e: Number,
    eta_tx: float = 0.95
) -> float:
    """
    電気エネルギー推定（単位：J）
    """
    # 入力検証
    if not (torque.shape == omega.shape and eta_tx > 0 and dt > 0):
        raise ValueError("Invalid input shape or non-positive dt/eta_tx")

    # 機械パワー
    P_mech = torque * omega

    # 逆起電力と電流
    V_emf = k_e * omega
    I = torque / k_e  # k_t == k_e 近似

    # 電気パワー（銅損＋電力変換）
    P_elec = R * I**2 + V_emf * I

    # 伝達効率を考慮した入力パワー
    P_in = P_elec / eta_tx

    # エネルギー積算
    return float(np.sum(P_in * dt))