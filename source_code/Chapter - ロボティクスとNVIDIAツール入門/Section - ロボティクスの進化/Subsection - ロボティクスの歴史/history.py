#!/usr/bin/env python3
"""
状態空間サイズ計算を含む
プロダクション対応ロボットカタログ
ROS 2ノードとしても動作可能。
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import List

# ROS 2 (rclpy) 対応
try:
    import rclpy
    from rclpy.node import Node
    ROS2_AVAILABLE = True
except ModuleNotFoundError:
    ROS2_AVAILABLE = False

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("robot_catalog")


@dataclass(frozen=True, slots=True)
class Robot:
    """ロボット仕様を不変オブジェクトとして保持"""
    name: str
    year: int
    dof: int
    actuation: str

    def state_size(self) -> int:
        """状態ベクトル長 = 関節角 + 関節速度"""
        return 2 * self.dof


class RobotCatalog:
    """ロボットデータベース"""

    def __init__(self, robots: List[Robot]) -> None:
        self._robots = robots

    @classmethod
    def default_catalog(cls) -> "RobotCatalog":
        return cls(
            [
                Robot("WABOT-1", 1973, 17, "electric"),
                Robot("Unimate", 1961, 6, "electric"),
                Robot("Honda_ASIMO", 2000, 34, "electric"),
                Robot("Boston_Atlas", 2013, 28, "hydraulic"),
            ]
        )

    def summary(self) -> None:
        """標準出力に状態サイズを出力"""
        for r in self._robots:
            logger.info("%s (%d): state_size=%d", r.name, r.year, r.state_size())


# ROS 2ノード化
if ROS2_AVAILABLE:

    class RobotCatalogNode(Node):
        def __init__(self) -> None:
            super().__init__("robot_catalog")
            self.catalog = RobotCatalog.default_catalog()
            self.timer = self.create_timer(1.0, self.publish_summary)

        def publish_summary(self) -> None:
            self.catalog.summary()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Robot catalog state-size calculator")
    parser.add_argument(
        "--ros2", action="store_true", help="ROS 2ノードとして実行"
    )
    args = parser.parse_args(argv)

    if args.ros2 and ROS2_AVAILABLE:
        rclpy.init()
        node = RobotCatalogNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        RobotCatalog.default_catalog().summary()


if __name__ == "__main__":
    main()