#!/usr/bin/env python3
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Isaac Sim
import carb
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.nucleus import get_assets_root_path
from pxr import UsdGeom, Gf

# 自作ユーティリティ（同ディレクトリに配置）
from utils.record_io import RecordWriter
from utils.randomizer import ScenarioRandomizer
from utils.policy import PolicyBase, ScriptedPolicy

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("data_gen")


class DataGenNode(Node):
    """ROS 2ノードとして動作するIsaac Simデータ収集器"""

    def __init__(self, config_path: str) -> None:
        super().__init__("isaac_data_gen")

        # パラメータ読み込み
        with open(config_path, "r") as f:
            self.cfg: Dict[str, Any] = json.load(f)

        # Isaac Sim起動
        self.sim_app = SimulationApp({"headless": self.cfg["headless"]})
        self.world = World(physics_dt=self.cfg["physics_dt"])

        # シーン読み込み
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise RuntimeError("Nucleusサーバに接続できません")
        stage_path = assets_root_path + self.cfg["stage_path"]
        open_stage(stage_path)
        self.world.reset()

        # Humanoidスポーン
        humanoid_prim = define_prim("/World/Humanoid", "Xform")
        # ここにUSDパスを解決してロードする実装を追加
        # usd_path = ...
        # humanoid_prim.GetReferences().AddReference(usd_path)

        # ランダマイザ・ポリシー・ライター初期化
        self.randomizer = ScenarioRandomizer(self.cfg["randomizer"])
        policy_cls = ScriptedPolicy if self.cfg["use_scripted_policy"] else PolicyBase
        self.policy = policy_cls(self.cfg["policy"])
        output_dir = Path(self.cfg["output_dir"]).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = RecordWriter(output_dir, self.cfg["record"])

        # タイマー登録（ROS 2スピン内で実行）
        self.timer = self.create_timer(
            self.cfg["physics_dt"], self.step_callback
        )

        # シードループ制御用
        self.seed_idx = 0
        self.t = 0
        self.steps_per_seed = self.cfg["steps_per_seed"]
        self.num_seeds = self.cfg["num_seeds"]

    def step_callback(self) -> None:
        """ROS 2タイマーコールバック：1ステップ進める"""
        if self.seed_idx >= self.num_seeds:
            self.finalize()
            return

        # シード切り替え
        if self.t == 0:
            seed = self.cfg["base_seed"] + self.seed_idx
            np.random.seed(seed)
            params = self.randomizer.sample(seed)
            self.world.reset()
            logger.info(f"Seed {seed} 開始")

        # 行動決定・ステップ
        action = self.policy.get_action(self.t, params)
        self.world.step(render=not self.cfg["headless"])

        # 観測取得
        rgb = self.capture_rgb()
        depth = self.capture_depth()
        seg = self.capture_segmentation()
        imu = self.read_imu()
        joints = self.read_joint_states()
        contacts = self.read_contact_forces()

        # 記録
        self.writer.write(
            seed=self.cfg["base_seed"] + self.seed_idx,
            step=self.t,
            rgb=rgb,
            depth=depth,
            seg=seg,
            imu=imu,
            joints=joints,
            contacts=contacts,
            meta=params,
        )

        self.t += 1
        if self.t >= self.steps_per_seed:
            self.seed_idx += 1
            self.t = 0

    def capture_rgb(self) -> np.ndarray:
        # ここにViewportからRGB取得
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def capture_depth(self) -> np.ndarray:
        return np.zeros((480, 640), dtype=np.float32)

    def capture_segmentation(self) -> np.ndarray:
        return np.zeros((480, 640), dtype=np.uint8)

    def read_imu(self) -> Dict[str, np.ndarray]:
        return {"accel": np.zeros(3), "gyro": np.zeros(3)}

    def read_joint_states(self) -> Dict[str, np.ndarray]:
        return {"pos": np.zeros(23), "vel": np.zeros(23)}

    def read_contact_forces(self) -> np.ndarray:
        return np.zeros(4)

    def finalize(self) -> None:
        logger.info("収集完了")
        self.writer.close()
        self.sim_app.close()
        rclpy.shutdown()


def main():
    rclpy.init()
    node = DataGenNode(
        config_path=os.path.join(
            os.path.dirname(__file__), "config", "data_gen.json"
        )
    )
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()


if __name__ == "__main__":
    main()