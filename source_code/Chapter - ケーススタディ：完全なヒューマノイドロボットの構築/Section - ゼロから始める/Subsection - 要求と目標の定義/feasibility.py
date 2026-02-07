#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Final

# 定数定義
P_AVG: Final[float] = 250.0          # W: 平均消費電力
T_MISSION: Final[float] = 3600.0     # s: ミッション時間
ETA_SYS: Final[float] = 0.85         # システム効率
E_DENSITY: Final[float] = 200.0      # Wh/kg: 使用可能エネルギー密度

I_LINK: Final[float] = 0.02          # kg·m²: リンク慣性モーメント
ALPHA: Final[float] = 2.0            # rad/s²: 要求加速度
M_LINK: Final[float] = 3.0           # kg: リンク質量
L_CM: Final[float] = 0.12            # m: 重心長さ
G: Final[float] = 9.81               # m/s²: 重力加速度
SAFETY_FACTOR: Final[float] = 2.0    # 安全率
MOTOR_TAU: Final[float] = 50.0       # Nm: モータ最大トルク


def compute_required_energy(p_avg: float, t_mission: float, eta_sys: float) -> float:
    """必要なバッテリエネルギー[Wh]を算出"""
    return p_avg * (t_mission / 3600.0) / eta_sys


def compute_battery_mass(e_req_wh: float, e_density: float) -> float:
    """バッテリ質量[kg]を算出"""
    if e_density <= 0:
        raise ValueError("e_density must be positive")
    return e_req_wh / e_density


def compute_required_torque(
    inertia: float, alpha: float, mass: float, length: float, theta: float = 0.0
) -> float:
    """必要な関節トルク[Nm]を算出（重力補償込み）"""
    return inertia * alpha + mass * G * length * math.cos(theta)


def main() -> None:
    # エネルギー・バッテリ質量算出
    e_req_wh = compute_required_energy(P_AVG, T_MISSION, ETA_SYS)
    battery_mass = compute_battery_mass(e_req_wh, E_DENSITY)

    # トルクチェック
    tau_req = compute_required_torque(I_LINK, ALPHA, M_LINK, L_CM)
    tau_with_sf = tau_req * SAFETY_FACTOR
    torque_ok = MOTOR_TAU >= tau_with_sf

    # 結果表示
    print(f"E_req_Wh={e_req_wh:.1f} Wh, battery_mass={battery_mass:.2f} kg")
    print(f"tau_req={tau_req:.2f} Nm, required_with_sf={tau_with_sf:.2f} Nm")
    print(f"torque_OK={torque_ok}")


if __name__ == "__main__":
    main()