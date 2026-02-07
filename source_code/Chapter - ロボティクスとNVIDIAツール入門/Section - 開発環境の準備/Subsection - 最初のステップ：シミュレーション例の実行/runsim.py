# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import carb
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.kit import SimulationApp

# ログレベル設定（Isaac Sim標準）
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HumanoidPD")

# 実行時設定
HUMANOID_USD: Optional[str] = os.environ.get("HUMANOID_USD_PATH")  # 環境変数優先
if HUMANOID_USD is None:
    # Nucleusにサンプルがあれば利用（存在チェック）
    assets_root = get_assets_root_path()
    if assets_root:
        HUMANOID_USD = str(Path(assets_root) / "Isaac/Robots/Humanoid/humanoid.usd")
    else:
        HUMANOID_USD = "/isaac-sim/assets/humanoid.usd"  # フォールバック

SIM_OPTIONS = {"headless": False, "width": 1280, "height": 720}
simulation_app = SimulationApp(SIM_OPTIONS)

# World初期化
world = World(stage_units_in_meters=1.0)
stage_utils.add_reference_to_stage(HUMANOID_USD, "/World/Humanoid")
robot = world.scene.add(
    Articulation(
        prim_path="/World/Humanoid",
        name="humanoid",
        position=np.array([0, 0, 1.0]),  # 地面にめり込み防止
    )
)

world.reset()
dt = 1.0 / 120.0
world.set_physics_dt(dt)

# PDゲイン（自動チューニングは省略、保守的に設定）
joint_indices = robot.get_active_joints()
dof = robot.num_dof
Kp = np.full(dof, 50.0)
Kd = np.full(dof, 1.0)

# 中立姿勢（USDのdefault位置を初期目標とする）
q_ref = robot.get_joint_positions()

# 制御ループ
steps = int(10.0 / dt)
log_interval = int(1.0 / (dt * 10))

for i in range(steps):
    world.step(render=True)
    q = robot.get_joint_positions()
    dq = robot.get_joint_velocities()
    tau = Kp * (q_ref - q) - Kd * dq  # 要素積で高速化
    robot.set_joint_efforts(tau)

    if i % log_interval == 0:
        com, _ = robot.get_world_pose()
        logger.info(f"step={i:05d}  COM_z={com[2]:.3f}")

simulation_app.close()