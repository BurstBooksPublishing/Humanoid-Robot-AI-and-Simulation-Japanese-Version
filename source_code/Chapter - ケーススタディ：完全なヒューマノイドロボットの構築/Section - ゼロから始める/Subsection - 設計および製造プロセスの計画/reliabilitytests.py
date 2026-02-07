"""
Production-ready reliability & integration estimator
ROS 2ノードとしても動作可能（rclpy対応）
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32


class ReliabilityCalculator(Node):
    """システム信頼度と統合試験数を計算するROS 2ノード"""

    def __init__(self) -> None:
        super().__init__("reliability_calculator")
        self._declare_parameters()
        self._setup_publishers()

    # -------------------------
    # パラメータ管理
    # -------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter("modules", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("reliabilities", rclpy.Parameter.Type.DOUBLE_ARRAY)

    # -------------------------
    # ROS 2 通信
    # -------------------------
    def _setup_publishers(self) -> None:
        self.sys_rel_pub = self.create_publisher(Float32, "~/system_reliability", 10)
        self.integ_cnt_pub = self.create_publisher(Int32, "~/integration_count", 10)
        self.timer = self.create_timer(1.0, self._publish_results)

    # -------------------------
    # 計算ロジック
    # -------------------------
    @staticmethod
    def system_reliability(reliabilities: Iterable[float]) -> float:
        """直列構成のシステム信頼度を返す（各要素の積）"""
        r = 1.0
        for ri in reliabilities:
            if not (0.0 <= ri <= 1.0):
                raise ValueError("信頼度は0〜1の範囲である必要があります")
            r *= ri
        return r

    @staticmethod
    def pairwise_tests(n: int) -> int:
        """完全グラフの辺数＝ペアワイズ統合試験数"""
        if n < 0:
            raise ValueError("モジュール数は非負でなければならない")
        return n * (n - 1) // 2

    # -------------------------
    # ROS 2 出力
    # -------------------------
    def _publish_results(self) -> None:
        modules = (
            self.get_parameter("modules").get_parameter_value().string_array_value
        )
        reliabilities = (
            self.get_parameter("reliabilities").get_parameter_value().double_array_value
        )

        if len(modules) != len(reliabilities):
            self.get_logger().error("モジュール数と信頼度数が一致しない")
            return

        sys_rel = self.system_reliability(reliabilities)
        integ_cnt = self.pairwise_tests(len(modules))

        self.sys_rel_pub.publish(Float32(data=sys_rel))
        self.integ_cnt_pub.publish(Int32(data=integ_cnt))


# -------------------------
# スタンドアロン実行
# -------------------------
def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = ReliabilityCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    # ROS 2未使用時の簡易実行
    try:
        rclpy
    except NameError:
        # ROS 2環境なし → 直接計算
        modules: Dict[str, float] = {
            "mechanical": 0.99,
            "actuation": 0.98,
            "sensing": 0.97,
            "control": 0.995,
        }
        calc = ReliabilityCalculator
        print(calc.system_reliability(modules.values()))
        print(calc.pairwise_tests(len(modules)))
    else:
        main()