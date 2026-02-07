# Isaac Sim 用の高品質サンプル（ROS 2 非使用）
import os
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.kit import SimulationApp

# 必要な拡張を有効化
enable_extension("omni.isaac.core")

# シグナルハンドラ：Ctrl-C でクリーン終了
def _sigint_handler(sig, frame):
    print("\n[INFO] 中断を検出。シミュレータを終了します。")
    SimulationApp.get_app().close()
    sys.exit(0)

signal.signal(signal.SIGINT, _sigint_handler)

# 起動オプション
CONFIG = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "sync_loads": True,
    "renderer": "RayTracedLighting",
}

# シミュレーション設定
SIM_DT = 1.0 / 500.0  # 500 Hz
SOLVER_ITER = 8
USD_PATH = Path("/path/to/humanoid.usd")  # 実ファイルを指定
ROBOT_PRIM = "/World/humanoid"
HIP_JOINT = "hip_joint"
FOOT_LINK = "right_foot"

# 軌道設定
AMP = 0.3  # rad
FREQ = 0.5  # Hz
DURATION = 2.0  # sec


def load_robot(world: World, usd_path: Path, prim_path: str) -> Articulation:
    """USD をステージに追加して Articulation オブジェクトを返す"""
    if not usd_path.exists():
        raise FileNotFoundError(f"USD not found: {usd_path}")
    world.scene.add_reference_to_stage(str(usd_path), prim_path)
    world.reset()  # 読み込み後に必ず reset
    return Articulation(prim_path)


def configure_articulation(art: Articulation) -> None:
    """関節ドライバ設定一括適用"""
    art.set_enabled_self_collisions(False)
    art.apply_drive_settings(
        {
            "leg.*": {"stiffness": 100.0, "damping": 10.0},
            "arm.*": {"stiffness": 80.0, "damping": 8.0},
        }
    )


def main() -> None:
    kit = SimulationApp(CONFIG)
    world = World()

    # 物理設定
    world.physics_context.set_time_step(SIM_DT)
    world.physics_context.get_physics_view().set_solver_iterations(SOLVER_ITER)

    # ロボット読み込み
    robot = load_robot(world, USD_PATH, ROBOT_PRIM)
    configure_articulation(robot)

    # ログ用バッファ
    log: List[Dict[str, float]] = []

    steps = int(DURATION / SIM_DT)
    for i in range(steps):
        t = i * SIM_DT
        q_des = AMP * np.sin(2.0 * np.pi * FREQ * t)
        robot.set_joint_position(HIP_JOINT, q_des)

        world.step(render=True)

        # 計測
        tau = robot.get_joint_effort(HIP_JOINT) or 0.0
        forces = robot.get_contact_forces(FOOT_LINK)
        force_mag = np.linalg.norm(forces) if forces is not None else 0.0
        log.append({"t": t, "tau": tau, "force": force_mag})

    # 簡易ログ出力
    for row in log[::50]:  # 10 ms 間隔で出力
        print(f"{row['t']:.3f}, torque={row['tau']:.3f}, contact={row['force']:.3f}")

    kit.close()


if __name__ == "__main__":
    main()