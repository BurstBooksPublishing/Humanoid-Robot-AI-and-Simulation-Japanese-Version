#!/usr/bin/env python3
"""
単リンクアームの必要トルクと自然振動数を計算するプロダクションスクリプト
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import yaml

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinkParams:
    """リンクパラメータを不変に保持"""
    mass: float          # kg
    com_distance: float  # m
    length: float        # m
    peak_accel: float    # rad/s^2


@dataclass(frozen=True)
class MotorParams:
    """モータ仕様"""
    nominal_torque: float  # Nm (連続)
    peak_torque: float     # Nm (瞬間)
    gear_ratio: float


@dataclass(frozen=True)
class JointParams:
    """関節特性"""
    stiffness: float  # Nm/rad


# 定数
GRAVITY: Final[float] = 9.81  # m/s^2


def compute_required_torque(link: LinkParams) -> float:
    """
    必要トルクを計算
    """
    # スランダロッド近似による慣性モーメント
    inertia = (1.0 / 3.0) * link.mass * link.length ** 2

    # 重力トルク
    tau_gravity = link.mass * GRAVITY * link.com_distance

    # 動的トルク（加速度項）
    tau_dynamic = (inertia + link.mass * link.com_distance ** 2) * link.peak_accel

    return tau_dynamic + tau_gravity


def compute_natural_frequency(link: LinkParams, joint: JointParams) -> float:
    """
    関節の自然振動数を計算
    """
    inertia = (1.0 / 3.0) * link.mass * link.length ** 2
    j_eff = inertia + link.mass * link.com_distance ** 2
    return np.sqrt(joint.stiffness / j_eff)


def compute_margin(required: float, selected: float) -> float:
    """
    トルクマージンを計算
    """
    if required <= 0:
        raise ValueError("required torque must be positive")
    return (selected - required) / required


def load_config(path: Path) -> dict:
    """
    YAML/JSON設定ファイルを読み込む
    """
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        return json.load(f)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="単リンクアームの必要トルク計算")
    parser.add_argument(
        "-c", "--config", type=Path, help="YAML/JSON設定ファイル"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="結果をJSONで出力するファイル"
    )
    args = parser.parse_args(argv)

    if args.config:
        cfg = load_config(args.config)
        link = LinkParams(**cfg["link"])
        motor = MotorParams(**cfg["motor"])
        joint = JointParams(**cfg["joint"])
    else:
        # デフォルト値
        link = LinkParams(mass=2.5, com_distance=0.25, length=0.5, peak_accel=10.0)
        motor = MotorParams(nominal_torque=1.2, peak_torque=3.6, gear_ratio=50.0)
        joint = JointParams(stiffness=500.0)

    tau_req = compute_required_torque(link)
    omega_n = compute_natural_frequency(link, joint)
    tau_selected = motor.peak_torque * motor.gear_ratio
    margin = compute_margin(tau_req, tau_selected)

    result = {
        "required_torque_Nm": round(tau_req, 2),
        "selected_torque_Nm": round(tau_selected, 1),
        "torque_margin": round(margin, 2),
        "natural_frequency_rad/s": round(omega_n, 2),
    }

    logger.info("tau_req=%.2f Nm, tau_selected=%.1f Nm, margin=%.2f", tau_req, tau_selected, margin)
    logger.info("natural_freq=%.2f rad/s", omega_n)

    if args.output:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("結果を %s に保存しました", args.output)


if __name__ == "__main__":
    main()