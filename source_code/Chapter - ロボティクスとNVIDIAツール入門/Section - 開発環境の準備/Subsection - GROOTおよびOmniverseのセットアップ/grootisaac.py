import os
import time
import logging
from pathlib import Path
from typing import Dict, Any

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

import groot
import carb
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Usd, UsdGeom

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HumanoidBTDriver")


class HumanoidBTDriver(Node):
    """ROS 2ノードとしてBTを駆動し、Isaac Simと同期"""

    def __init__(self) -> None:
        super().__init__("humanoid_bt_driver")

        # パラメータ宣言
        self.declare_parameter(
            "behavior_tree_path",
            "",
            ParameterDescriptor(
                description="GROOT BTファイルの絶対パス",
                type=ParameterType.PARAMETER_STRING,
            ),
        )
        self.declare_parameter(
            "humanoid_prim_path",
            "/World/Humanoid",
            ParameterDescriptor(
                description="HumanoidのUSD Primパス",
                type=ParameterType.PARAMETER_STRING,
            ),
        )
        self.declare_parameter(
            "simulation_freq",
            120.0,
            ParameterDescriptor(
                description="Isaac Sim物理ステップ周波数[Hz]",
                type=ParameterType.PARAMETER_DOUBLE,
            ),
        )
        self.declare_parameter(
            "bt_tick_freq",
            30.0,
            ParameterDescriptor(
                description="BT tick周波数[Hz]",
                type=ParameterType.PARAMETER_DOUBLE,
            ),
        )

        # パラメータ取得
        bt_path = (
            self.get_parameter("behavior_tree_path")
            .get_parameter_value()
            .string_value
        )
        if not bt_path or not Path(bt_path).is_file():
            raise FileNotFoundError(f"BTファイルが見つかりません: {bt_path}")

        humanoid_path = (
            self.get_parameter("humanoid_prim_path")
            .get_parameter_value()
            .string_value
        )
        sim_freq = (
            self.get_parameter("simulation_freq").get_parameter_value().double_value
        )
        bt_freq = (
            self.get_parameter("bt_tick_freq").get_parameter_value().double_value
        )

        # World取得
        self.world = World.instance()
        stage = self.world.stage
        self.humanoid_prim = get_prim_at_path(humanoid_path)
        if not self.humanoid_prim.IsValid():
            raise RuntimeError(f"Primが見つかりません: {humanoid_path}")

        # BT読み込み
        self.bt = groot.load_tree(bt_path)
        self.bt.register_action("ApplyJointTargets", self.apply_joint_targets)

        # 周期設定
        self.dt_sim = 1.0 / sim_freq
        self.dt_bt = 1.0 / bt_freq
        self.next_bt_time = self.world.current_time + self.dt_bt

        # タイマー作成（BT tick用）
        self.create_timer(self.dt_bt, self.timer_callback)

        logger.info("HumanoidBTDriver初期化完了")

    def apply_joint_targets(self, bb: Dict[str, Any]) -> groot.Status:
        """ブラックボードから関節目標を取得しIsaacへ適用"""
        targets = bb.get("joint_targets")
        if targets is None:
            logger.warning("joint_targetsがブラックボードに存在しません")
            return groot.Status.FAILURE

        # ArticulationKernelへ送信
        robot = self.world.scene.get_object(self.humanoid_prim.GetName())
        if isinstance(robot, Robot):
            robot.set_joint_positions(targets)
            return groot.Status.SUCCESS
        logger.error("Robotオブジェクトが取得できません")
        return groot.Status.FAILURE

    def timer_callback(self) -> None:
        """BT tickタイマー"""
        self.bt.tick()


def main(args=None):
    rclpy.init(args=args)
    node = HumanoidBTDriver()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()