"""
Production-ready battery sizing utility for humanoid robots.
ROS 2ノードとしても動作可能（rclpy対応）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

# ROS 2利用時のみインポートを試みる
try:
    import rclpy
    from rclpy.node import Node
    ROS2_AVAILABLE: Final[bool] = True
except ModuleNotFoundError:
    ROS2_AVAILABLE = False


# 定数は型付きで一元管理
class BatteryConsts:
    """リチウムイオン電池に関する定数"""
    LI_ION_NOMINAL_V: Final[float] = 3.7  # V
    MIN_CELL_V: Final[float] = 3.0       # 過放電保護閾値
    MAX_CELL_V: Final[float] = 4.2       # 満充電時


@dataclass(slots=True, frozen=True)
class BatterySizingResult:
    """計算結果を不変オブジェクトで返す"""
    c_required_ah: float
    i_peak_a: float
    implied_c: float
    v_term_peak: float
    cells_series: int
    can_handle_peak: bool


def battery_size(
    p_avg_w: float,
    p_peak_w: float,
    t_hours: float,
    v_nom_v: float,
    eta_sys: float,
    cell_capacity_ah: float,
    cell_c_rate: float,
    r_int_per_pack_ohm: float,
) -> BatterySizingResult:
    """
    必要なパック容量とセル構成を推定する。
    負値・ゼロ除算は ValueError で早期リジェクト。
    """
    # 入力値の簡易バリデーション
    if p_avg_w <= 0 or p_peak_w <= 0 or t_hours <= 0 or v_nom_v <= 0:
        raise ValueError("平均・ピーク電力、時間、電圧は正でなければならない")
    if not (0 < eta_sys <= 1):
        raise ValueError("システム効率は 0 < η ≤ 1")
    if cell_capacity_ah <= 0 or cell_c_rate <= 0 or r_int_per_pack_ohm < 0:
        raise ValueError("セル仕様が不正")

    # 必要容量（有効容量÷効率）
    c_req = (p_avg_w * t_hours) / (v_nom_v * eta_sys)

    # ピーク電流
    i_peak = p_peak_w / v_nom_v

    # パック全体としての暗黙Cレート
    implied_c = i_peak / c_req if c_req > 0 else math.inf

    # ピーク負荷時の終端電圧（実効電圧）
    v_term = v_nom_v - i_peak * r_int_per_pack_ohm

    # 直列セル数（切り上げで安全側に）
    cells_in_series = max(1, math.ceil(v_nom_v / BatteryConsts.LI_ION_NOMINAL_V))

    # ピーク電流をセルCレートで満たせるか
    can_handle_peak = (cell_c_rate * cell_capacity_ah) >= i_peak

    return BatterySizingResult(
        c_required_ah=c_req,
        i_peak_a=i_peak,
        implied_c=implied_c,
        v_term_peak=v_term,
        cells_series=cells_in_series,
        can_handle_peak=can_handle_peak,
    )


# ------------------------------------------------------------------
# ROS 2ノード化（オプション）
# ------------------------------------------------------------------
if ROS2_AVAILABLE:

    class BatterySizingNode(Node):
        def __init__(self) -> None:
            super().__init__("battery_sizer")
            self.declare_parameters(
                namespace="",
                parameters=[
                    ("p_avg_w", 200.0),
                    ("p_peak_w", 1500.0),
                    ("t_hours", 2.0),
                    ("v_nom_v", 48.0),
                    ("eta_sys", 0.9),
                    ("cell_capacity_ah", 3.0),
                    ("cell_c_rate", 10.0),
                    ("r_int_per_pack_ohm", 0.05),
                ],
            )
            self.timer = self.create_timer(1.0, self.run_sizing)

        def run_sizing(self) -> None:
            p_avg = self.get_parameter("p_avg_w").value
            p_peak = self.get_parameter("p_peak_w").value
            t_hours = self.get_parameter("t_hours").value
            v_nom = self.get_parameter("v_nom_v").value
            eta = self.get_parameter("eta_sys").value
            cap = self.get_parameter("cell_capacity_ah").value
            crate = self.get_parameter("cell_c_rate").value
            r_int = self.get_parameter("r_int_per_pack_ohm").value

            try:
                result = battery_size(p_avg, p_peak, t_hours, v_nom, eta, cap, crate, r_int)
                self.get_logger().info(f"{result}")
            except ValueError as e:
                self.get_logger().error(f"Invalid params: {e}")


# ------------------------------------------------------------------
# 単体実行（python battery_sizer.py）
# ------------------------------------------------------------------
if __name__ == "__main__":
    if ROS2_AVAILABLE and rclpy.ok():
        rclpy.init()
        node = BatterySizingNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()
    else:
        # スタンドアロンテスト
        result = battery_size(
            p_avg_w=200,
            p_peak_w=1500,
            t_hours=2,
            v_nom_v=48,
            eta_sys=0.9,
            cell_capacity_ah=3.0,
            cell_c_rate=10,
            r_int_per_pack_ohm=0.05,
        )
        print(result)