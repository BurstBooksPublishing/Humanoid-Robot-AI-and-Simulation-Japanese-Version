"""
厚さ推薦モジュール
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

# 材料定数（必要に応じて外部化）
E_ALUMINUM: Final[float] = 69e9          # [Pa]
E_NYLON: Final[float] = 3e9              # [Pa]
E_SLA_RESIN: Final[float] = 2.5e9        # [Pa]

# 製造方法判定閾値（単位：m）
THRESH_CNC: Final[float] = 5e-3          # 5 mm
TOL_CNC: Final[float] = 0.05e-3          # 0.05 mm
TOL_SLS_FDM: Final[float] = 0.2e-3       # 0.2 mm


@dataclass(frozen=True)
class Recommendation:
    """計算結果と製造方法を保持"""
    thickness_m: float
    method: str


def recommend_thickness(
    force_N: float,
    length_m: float,
    width_m: float,
    young_mod_Pa: float,
    delta_max_m: float,
    tolerance_m: float,
) -> Recommendation:
    """
    ビームのたわみ制約を満たす最小板厚と製造方法を返す。
    たわみ公式：δ = (F L³)/(3 E I) 、I = b t³/12 を仮定。
    """
    if any(v <= 0 for v in (force_N, length_m, width_m, young_mod_Pa, delta_max_m)):
        raise ValueError("全ての入力は正でなければならない")

    # 最小断面二次モーメントを逆算
    i_min = (force_N * length_m ** 3) / (3 * young_mod_Pa * delta_max_m)
    # 必要板厚
    t_min = ((12 * i_min) / width_m) ** (1.0 / 3.0)

    # 製造方法判定
    if t_min >= THRESH_CNC and tolerance_m <= TOL_CNC:
        method = "CNC_mill_aluminum"
    elif t_min < THRESH_CNC and tolerance_m <= TOL_SLS_FDM:
        method = "SLS_or_FDM_with_metal_inserts"
    else:
        method = "SLA_with_postcure_and_machined_features"

    return Recommendation(thickness_m=t_min, method=method)


# 使用例
if __name__ == "__main__":
    result = recommend_thickness(
        force_N=50.0,
        length_m=0.08,
        width_m=0.02,
        young_mod_Pa=E_NYLON,
        delta_max_m=1e-3,
        tolerance_m=0.2e-3,
    )
    print(f"min_thickness = {result.thickness_m:.4f} m, method = {result.method}")