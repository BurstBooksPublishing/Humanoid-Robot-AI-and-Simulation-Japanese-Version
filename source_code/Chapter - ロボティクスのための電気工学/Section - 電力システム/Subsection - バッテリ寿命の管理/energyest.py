#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import math
from typing import Final

# 充電効率のデフォルト値（リジェネ時）
DEFAULT_ETA_REG: Final[float] = 0.90
# SOC の許容範囲
SOC_MIN: Final[float] = 0.0
SOC_MAX: Final[float] = 1.0


def update_soc(
    soc: float,
    q_ah: float,
    i_a: float,
    dt_s: float,
    eta_reg: float = DEFAULT_ETA_REG,
) -> float:
    """
    クーロンカウンティングによる SOC 推定（リジェネ効率考慮）
    i_a > 0: 放電, i_a < 0: 回生
    """
    if q_ah <= 0:
        raise ValueError("q_ah must be positive")
    if dt_s < 0:
        raise ValueError("dt_s must be non-negative")

    # 変化量を Ah 単位で計算
    dah = i_a * dt_s / 3600.0
    if i_a < 0:
        dah *= eta_reg  # 回生時は充電効率を適用

    soc_new = soc - dah / q_ah
    # クランプ処理
    return max(SOC_MIN, min(SOC_MAX, soc_new))


def predict_runtime(
    soc: float,
    q_ah: float,
    v_nom: float,
    i_avg_a: float,
) -> float:
    """
    平均電流 i_avg_a で定常放電した場合の残り運転時間を秒単位で返す
    """
    if q_ah <= 0:
        raise ValueError("q_ah must be positive")
    if i_avg_a <= 0:
        return math.inf  # 放電電流がゼロ以下なら無限
    ah_rem = soc * q_ah
    return ah_rem / i_avg_a * 3600.0


# 動作確認
if __name__ == "__main__":
    soc = 0.9
    q_ah = 10.0  # 10 Ah パック
    v_nom = 25.9  # 今回は未使用だが将来的な拡張用に保持

    soc = update_soc(soc, q_ah, i_a=15.0, dt_s=0.5)  # 15 A バースト
    runtime_s = predict_runtime(soc, q_ah, v_nom, i_avg_a=8.0)
    print(f"SOC: {soc:.3f}, Runtime: {runtime_s:.1f} s")