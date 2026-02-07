import os
import signal
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import rclpy
from omni.isaac.core import World
from omni.isaac.kit import SimulationApp
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class ContactMonitor(Node):
    """Isaac Sim シミュレーション中の接触・関節情報を CSV に記録するノード"""

    def __init__(self, world: World, log_path: Path, robot_prim_path: str) -> None:
        super().__init__("contact_monitor")
        self.world = world
        self.robot_prim_path = robot_prim_path
        self.dt = 1.0 / 240.0  # 固定シミュレーション周期

        # CSV 初期化
        self.log_path = log_path
        self.csvfile = open(self.log_path, "w", newline="")
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow(
            ["sim_time", "max_penetration_m", "active_contacts",
             "max_joint_vel", "max_command_torque"]
        )
        self.csv_lock = threading.Lock()

        # 定期ログ用タイマー
        self.create_timer(1.0, self.timer_callback)

        # シャットダウンシグナル登録
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def step(self) -> None:
        """1ステップ分のデータ収集"""
        self.world.step(self.dt)
        self.world.update()
        sim_time = self.world.current_time

        # 接触情報取得
        contacts = self.world.get_active_contacts()
        max_pen = max((c["penetration_depth"] for c in contacts), default=0.0)
        active_contacts = len(contacts)

        # 関節情報取得
        joint_vels = self.world.get_joint_velocities(self.robot_prim_path)
        cmd_torques = self.world.get_commanded_torques(self.robot_prim_path)
        max_joint_vel = max(abs(v) for v in joint_vels) if joint_vels else 0.0
        max_cmd_torque = max(abs(t) for t in cmd_torques) if cmd_torques else 0.0

        # CSV 書き込み（排他制御）
        with self.csv_lock:
            self.writer.writerow(
                [sim_time, max_pen, active_contacts, max_joint_vel, max_cmd_torque]
            )

    def timer_callback(self) -> None:
        """1 Hz でログを画面出力"""
        contacts = self.world.get_active_contacts()
        max_pen = max((c["penetration_depth"] for c in contacts), default=0.0)
        self.get_logger().info(
            f"t={self.world.current_time:.2f}s "
            f"contacts={len(contacts)} max_pen={max_pen:.4f}m"
        )

    def _signal_handler(self, signum, frame) -> None:
        """シグナル受信時にクリーンアップ"""
        self.get_logger().info("Shutdown signal received.")
        self.shutdown()

    def shutdown(self) -> None:
        """リソース解放"""
        with self.csv_lock:
            self.csvfile.flush()
            self.csvfile.close()
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)

    # Isaac Sim 起動
    sim_app = SimulationApp({"headless": False})
    world = World(stage_units_in_meters=1.0)
    world.reset()

    # ログ保存先
    log_path = Path(os.environ.get("ISAAC_LOG_DIR", ".")) / "contact_debug.csv"

    # ノード作成
    monitor = ContactMonitor(world, log_path, robot_prim_path="/World/humanoid")

    try:
        while sim_app.is_running():
            monitor.step()
            time.sleep(0.0)  # headless 時は 0
    except KeyboardInterrupt:
        pass
    finally:
        monitor.shutdown()
        sim_app.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()