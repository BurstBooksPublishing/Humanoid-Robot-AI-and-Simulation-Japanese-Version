#!/usr/bin/env python3
"""
バッテリ・ケーブルサイズ計算（ヒューマノイドロボット用）
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

# 物理定数
RHO_CU: Final[float] = 1.68e-8  # 銅の抵抗率 [Ω·m]

@dataclass(frozen=True)
class MissionSpec:
    """ミッション仕様"""
    p_avg_w: float          # 平均消費電力 [W]
    duration_h: float       # 継続時間 [h]
    v_nom_v: float          # 公称バス電圧 [V]
    e_spec_wh_kg: float     # バッテリ質量エネルギー密度 [Wh/kg]
    l_wire_m: float         # 片方向ケーブル長 [m]
    i_peak_a: float         # ピーク電流 [A]
    v_drop_ratio: float     # 許容電圧降下率 [-]

@dataclass(frozen=True)
class SizingResult:
    """サイジング結果"""
    energy_wh: float
    capacity_ah: float
    battery_mass_kg: float
    min_area_mm2: float

def compute_battery_cable(spec: MissionSpec) -> SizingResult:
    """
    バッテリ容量と導体最小断面積を計算
    """
    energy_wh = spec.p_avg_w * spec.duration_h
    capacity_ah = energy_wh / spec.v_nom_v
    battery_mass_kg = energy_wh / spec.e_spec_wh_kg

    # 許容抵抗値から最小断面積を逆算
    r_max_ohm = (spec.v_drop_ratio * spec.v_nom_v) / spec.i_peak_a
    area_m2 = (RHO_CU * spec.l_wire_m) / r_max_ohm
    area_mm2 = area_m2 * 1e6

    return SizingResult(
        energy_wh=energy_wh,
        capacity_ah=capacity_ah,
        battery_mass_kg=battery_mass_kg,
        min_area_mm2=area_mm2,
    )

def main() -> None:
    spec = MissionSpec(
        p_avg_w=300.0,
        duration_h=1.0,
        v_nom_v=48.0,
        e_spec_wh_kg=200.0,
        l_wire_m=2.0,
        i_peak_a=30.0,
        v_drop_ratio=0.02,
    )

    res = compute_battery_cable(spec)

    print(f"Required energy: {res.energy_wh:.1f} Wh")
    print(f"Battery capacity: {res.capacity_ah:.2f} Ah at {spec.v_nom_v} V")
    print(f"Estimated battery mass: {res.battery_mass_kg:.2f} kg")
    print(f"Minimum conductor area: {res.min_area_mm2:.2f} mm²")

if __name__ == "__main__":
    main()