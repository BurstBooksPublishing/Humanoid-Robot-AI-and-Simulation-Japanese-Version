#!/usr/bin/env python3
import math
from typing import Tuple

# リンクパラメータ（CAD 値）
m1, m2 = 2.5, 1.8          # kg
l1, l2 = 0.24, 0.18        # m
I1, I2 = 0.012, 0.006      # kg·m²

# 所望動作
ALPHA_PEAK = 6.0           # rad/s²
THETA_HORZ = 0.0           # rad（水平姿勢＝重力トルク最大）

# 定数
g = 9.81                   # m/s²
SAFETY = 2.0               # 安全率
MOTOR_TAU_CONT = 0.8       # Nm（モータ連続定格トルク）


def calc_required_torque(m1: float, m2: float,
                         l1: float, l2: float,
                         I1: float, I2: float,
                         alpha: float, theta: float) -> Tuple[float, float]:
    """
    肩関節に必要な最大トルクを計算
    戻り値: (tau_grav, tau_dyn)
    """
    # 等価慣性モーメント（平行軸定理＋リンク2の遠端近似）
    I_eq = I1 + m1 * (l1 ** 2) + I2 + m2 * ((l1 + l2 / 2) ** 2)

    # 重力トルク（水平姿勢で最大）
    tau_grav = (m1 * g * l1 + m2 * g * (l1 + l2 / 2)) * math.cos(theta)

    # 動的トルク（慣性）
    tau_dyn = I_eq * alpha

    return tau_grav, tau_dyn


def recommend_gearbox(tau_total: float, motor_cont: float, safety: float) -> int:
    """
    必要トルクとモータ定格から最小ギア比を整数で返す
    """
    ratio = (safety * tau_total) / motor_cont
    return math.ceil(ratio)


def main() -> None:
    tau_grav, tau_dyn = calc_required_torque(
        m1, m2, l1, l2, I1, I2, ALPHA_PEAK, THETA_HORZ)
    tau_req = tau_grav + tau_dyn

    gear_ratio = recommend_gearbox(tau_req, MOTOR_TAU_CONT, SAFETY)

    print(f"必要トルク: {tau_req:.2f} Nm")
    print(f"推奨ギアボックス比: {gear_ratio} : 1")


if __name__ == "__main__":
    main()