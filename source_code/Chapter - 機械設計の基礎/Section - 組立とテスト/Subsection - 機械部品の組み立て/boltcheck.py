#!/usr/bin/env python3
"""
Production-ready bolt preload & shear-margin calculator
ROS 2ノードとしても動作可能（rclpy使用）
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


# ---------- 定数 ----------
# 鋼の代表強度（Pa）
STEEL_SHEAR_STRENGTH: Final[float] = 0.6e9  # 0.6 GPa


# ---------- データ構造 ----------
@dataclass(slots=True, frozen=True)
class BoltSpec:
    """ボルト仕様を不変保持"""
    T: float          # 締付トルク [N·m]
    d: float          # 公称径 [m]
    K: float          # トルク係数 [-]
    A_t: float        # 引張応力面積 [m²]
    A_s: float        # せん断応力面積 [m²]


@dataclass(slots=True, frozen=True)
class LoadCondition:
    """荷重条件"""
    M_r: float        # 伝達モーメント [N·m]
    r_eff: float      # 実効半径 [m]
    mu: float         # 摩擦係数 [-]


# ---------- 計算ロジック ----------
def calc_bolt_areas(d: float) -> tuple[float, float]:
    """引張・せん断面積を算出"""
    A_t = math.pi * (d / 2.0) ** 2
    A_s = 0.7854 * d ** 2 * 0.6
    return A_t, A_s


def evaluate_bolt(spec: BoltSpec, load: LoadCondition) -> tuple[float, float, float]:
    """
    プリロード・軸応力・せん断余裕を返す
    Returns:
        F_p [N], sigma_a [Pa], shear_margin [Pa]
    """
    F_p = spec.T / (spec.K * spec.d)                 # トルク則
    sigma_a = F_p / spec.A_t                         # 軸応力
    tau = (load.M_r / load.r_eff) / spec.A_s         # 発生せん断応力
    shear_margin = STEEL_SHEAR_STRENGTH - tau        # 余裕（絶対値）
    return F_p, sigma_a, shear_margin


# ---------- ROS 2 ノード ----------
class BoltCheckNode(Node):
    def __init__(self) -> None:
        super().__init__("bolt_check_node")
        self.pub = self.create_publisher(Float64MultiArray, "~/bolt_diagnostics", 10)
        timer_period: float = 1.0  # [s]
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # パラメータ読取
        self.declare_parameter("torque_Nm", 25.0)
        self.declare_parameter("diameter_m", 0.010)
        self.declare_parameter("torque_coeff", 0.2)
        self.declare_parameter("moment_Nm", 12.0)
        self.declare_parameter("radius_m", 0.015)
        self.declare_parameter("mu", 0.15)

    def timer_callback(self) -> None:
        T = self.get_parameter("torque_Nm").value
        d = self.get_parameter("diameter_m").value
        K = self.get_parameter("torque_coeff").value
        M_r = self.get_parameter("moment_Nm").value
        r_eff = self.get_parameter("radius_m").value
        mu = self.get_parameter("mu").value

        A_t, A_s = calc_bolt_areas(d)
        spec = BoltSpec(T=T, d=d, K=K, A_t=A_t, A_s=A_s)
        load = LoadCondition(M_r=M_r, r_eff=r_eff, mu=mu)

        F_p, sigma_a, margin = evaluate_bolt(spec, load)

        # ログ出力（日本語）
        self.get_logger().info(f"プリロード: {F_p:.1f} N")
        self.get_logger().info(f"軸応力: {sigma_a/1e6:.2f} MPa")
        self.get_logger().info(f"せん断余裕: {margin/1e6:.2f} MPa")

        # 診断データ配信
        msg = Float64MultiArray()
        msg.data = [F_p, sigma_a, margin]
        self.pub.publish(msg)


# ---------- スタンドアロン実行 ----------
def main_standalone() -> None:
    """ROS無しで単体実行"""
    # 初期値（股アクチュエータ例）
    T = 25.0
    d = 0.010
    K = 0.2
    M_r = 12.0
    r_eff = 0.015
    mu = 0.15

    A_t, A_s = calc_bolt_areas(d)
    spec = BoltSpec(T=T, d=d, K=K, A_t=A_t, A_s=A_s)
    load = LoadCondition(M_r=M_r, r_eff=r_eff, mu=mu)

    F_p, sigma_a, margin = evaluate_bolt(spec, load)

    print(f"# プリロード (N): {F_p}")
    print(f"# 軸応力 (MPa): {sigma_a/1e6}")
    print(f"# せん断応力 (MPa): {(M_r/r_eff)/A_s/1e6}")
    print(f"# せん断余裕 (MPa): {margin/1e6}")


def main() -> None:
    try:
        rclpy.init()
        node = BoltCheckNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    # ROS環境変数があればノード起動、なければスタンドアロン
    if "ROS_DOMAIN_ID" in __import__("os").environ:
        main()
    else:
        main_standalone()