import numpy as np
from typing import Tuple, Union

FloatArray = Union[np.ndarray, Tuple[float, float, float], list]

def accel_correct(a_com: FloatArray,
                  alpha: FloatArray,
                  omega: FloatArray,
                  r: FloatArray) -> np.ndarray:
    """
    加速度計のオフセット補正（重力補正なし）

    Parameters
    ----------
    a_com : 形状(3,)の加速度ベクトル [m/s^2]
    alpha : 形状(3,)の角加速度ベクトル [rad/s^2]
    omega : 形状(3,)の角速度ベクトル [rad/s]
    r     : 形状(3,)のIMUからセンサ原点へのオフセットベクトル [m]

    Returns
    -------
    np.ndarray
        補正済みセンサ加速度（重力なし）[m/s^2]
    """
    a_com   = np.asarray(a_com, dtype=np.float64).flatten()
    alpha   = np.asarray(alpha, dtype=np.float64).flatten()
    omega   = np.asarray(omega, dtype=np.float64).flatten()
    r       = np.asarray(r, dtype=np.float64).flatten()

    if any(v.size != 3 for v in (a_com, alpha, omega, r)):
        raise ValueError("全ての入力は3次元ベクトルである必要があります")

    coriolis   = np.cross(alpha, r)                     # コリオリ加速度
    centripetal = np.cross(omega, np.cross(omega, r))   # 遠心加速度
    a_sensor   = a_com + coriolis + centripetal         # 補正後加速度
    return a_sensor