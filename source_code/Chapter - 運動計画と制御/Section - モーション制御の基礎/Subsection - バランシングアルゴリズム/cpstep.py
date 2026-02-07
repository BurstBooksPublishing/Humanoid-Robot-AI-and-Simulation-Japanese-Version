#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from typing import Tuple, Final

# 物理定数
GRAVITY: Final[float] = 9.81  # [m/s^2]
COM_HEIGHT: Final[float] = 0.9  # [m]

# 歩行パラメータ
STEP_TIME: Final[float] = 0.6  # [s]
MAX_STEP_REACH: Final[float] = 0.4  # [m]

# 自然周波数
OMEGA: Final[float] = np.sqrt(GRAVITY / COM_HEIGHT)


def capture_point(com_pos: np.ndarray, com_vel: np.ndarray) -> np.ndarray:
    """
    キャプチャポイントを計算
    """
    return com_pos + com_vel / OMEGA


def propose_step(
    com_pos: np.ndarray,
    com_vel: np.ndarray,
    current_zmp: np.ndarray,
) -> np.ndarray:
    """
    次のZMP候補を提案（単一ステップ）
    """
    xi = capture_point(com_pos, com_vel)
    # 指数減衰モデルによる目標ZMP
    target_zmp = xi + np.exp(-OMEGA * STEP_TIME) * (xi - current_zmp)
    # 運動学的制約でクリップ
    delta = np.clip(target_zmp - current_zmp, -MAX_STEP_REACH, MAX_STEP_REACH)
    return current_zmp + delta


def main() -> None:
    # 初期状態
    com_pos = np.array([0.0, 0.0])      # [m]
    com_vel = np.array([0.3, 0.0])      # [m/s]
    current_zmp = np.array([0.0, 0.0])  # [m]

    next_zmp = propose_step(com_pos, com_vel, current_zmp)
    print(f"Proposed next ZMP: {next_zmp}")


if __name__ == "__main__":
    main()