#!/usr/bin/env python3
"""
Tolerance stack-up analyzer for encoder mounting interface.
Computes worst-case and RSS (Root-Sum-Square) tolerance stack-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, List

# 許容差仕様（mm）
@dataclass(frozen=True)
class ToleranceSpec:
    bearing_seat: float
    spacer: float
    shaft_runout: float

    def to_list(self) -> List[float]:
        return [self.bearing_seat, self.spacer, self.shaft_runout]

# 組立許容基準（mm）
MAX_RUNOUT: Final[float] = 0.15  # エンコーダ最大許容振れ
RSS_THRESHOLD_RATIO: Final[float] = 1.0 / 3.0  # RSS警告閾値比率

def compute_stack_up(spec: ToleranceSpec) -> tuple[float, float]:
    """
    最悪値とRSS積み重ねを計算
    Returns (worst_case, rss)
    """
    tolerances = spec.to_list()
    worst_case = sum(tolerances)
    rss = math.sqrt(sum(t * t for t in tolerances))
    return worst_case, rss

def evaluate_stack_up(worst_case: float, rss: float) -> None:
    """
    積み重ね結果を評価し必要に応じて警告
    """
    print(f"Worst-case: {worst_case:.3f} mm")
    print(f"RSS: {rss:.3f} mm")

    if rss > MAX_RUNOUT * RSS_THRESHOLD_RATIO:
        print("Action: tighten tolerances or add compliant coupling")

def main() -> None:
    # サプライヤデータに応じて調整
    spec = ToleranceSpec(bearing_seat=0.10, spacer=0.08, shaft_runout=0.12)
    worst_case, rss = compute_stack_up(spec)
    evaluate_stack_up(worst_case, rss)

if __name__ == "__main__":
    main()