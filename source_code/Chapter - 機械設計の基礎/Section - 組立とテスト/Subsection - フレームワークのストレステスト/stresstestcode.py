#!/usr/bin/env python3
import argparse
import csv
import logging
import os
import time
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np

# --------------------------------------------------------------------------- #
# ハードウェア抽象インターフェース（本番環境では実ドライバに置換）            #
# --------------------------------------------------------------------------- #

class ControllerInterface(Protocol):
    def enable_safety_lock(self) -> None: ...
    def disable_safety_lock(self) -> None: ...
    def set_torque(self, joint_id: int, torque: float) -> None: ...
    def state(self) -> "JointState": ...

class SensorInterface(Protocol):
    def read_strain(self) -> np.ndarray: ...
    def read_force(self) -> np.ndarray: ...

@dataclass
class JointState:
    pos: float
    vel: float
    eff: float

# --------------------------------------------------------------------------- #
# 本番用実装                                                                   #
# --------------------------------------------------------------------------- #

class TorqueRampConfig:
    def __init__(self) -> None:
        self.torque_max: float = 50.0  # Nm
        self.torque_step: float = 1.0  # Nm/sec
        self.strain_threshold: float = 2000e-6  # micro-strain
        self.dt: float = 0.2  # 指令更新周期
        self.joint_id: int = 2
        self.output_dir: Path = Path(os.environ.get("RAMP_LOG_DIR", "."))

class TorqueRampTest:
    def __init__(
        self,
        controller: ControllerInterface,
        sensors: SensorInterface,
        cfg: TorqueRampConfig,
    ) -> None:
        self.ctrl = controller
        self.sns = sensors
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

    # 緊急停止
    def _emergency_stop(self, joint_id: int) -> None:
        self.ctrl.set_torque(joint_id, 0.0)
        self.ctrl.disable_safety_lock()
        self.logger.error("緊急停止：ひずみ閾値超過")

    # ログ書き出し
    def _log_sample(self, writer: csv.writer, torque: float) -> None:
        strain = self.sns.read_strain()
        force = self.sns.read_force()
        state = self.ctrl.state()
        writer.writerow(
            [
                time.time(),
                torque,
                *strain,
                *force,
                state.pos,
                state.vel,
                state.eff,
            ]
        )

    # メインループ
    def run(self) -> None:
        self.ctrl.enable_safety_lock()
        torque = 0.0
        log_path = self.cfg.output_dir / f"torque_ramp_{int(time.time())}.csv"
        with log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "command_torque"]
                + [f"strain_{i}" for i in range(len(self.sns.read_strain()))]
                + [f"force_{i}" for i in range(len(self.sns.read_force()))]
                + ["pos", "vel", "eff"]
            )
            while torque <= self.cfg.torque_max:
                self.ctrl.set_torque(self.cfg.joint_id, torque)
                time.sleep(self.cfg.dt)
                self._log_sample(writer, torque)
                if np.any(np.abs(self.sns.read_strain()) > self.cfg.strain_threshold):
                    self._emergency_stop(self.cfg.joint_id)
                    raise RuntimeError("Strain threshold exceeded")
                torque += self.cfg.torque_step
        self.ctrl.set_torque(self.cfg.joint_id, 0.0)
        self.ctrl.disable_safety_lock()
        self.logger.info(f"試験完了：ログ={log_path}")

# --------------------------------------------------------------------------- #
# ハードウェアスタブ（本番では削除）                                           #
# --------------------------------------------------------------------------- #

class FakeController:
    def enable_safety_lock(self) -> None: pass
    def disable_safety_lock(self) -> None: pass
    def set_torque(self, _: int, __: float) -> None: pass
    def state(self) -> JointState: return JointState(0.0, 0.0, 0.0)

class FakeSensors:
    def read_strain(self) -> np.ndarray: return np.zeros(4)
    def read_force(self) -> np.ndarray: return np.zeros(3)

# --------------------------------------------------------------------------- #
# エントリポイント                                                             #
# --------------------------------------------------------------------------- #

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", type=int, default=2, help="対象ジョイントID")
    parser.add_argument("--max-torque", type=float, default=50.0, help="最大トルク[Nm]")
    parser.add_argument("--step", type=float, default=1.0, help="トルク増分[Nm]")
    parser.add_argument("--strain-limit", type=float, default=2000e-6, help="ひずみ閾値")
    parser.add_argument("--log-dir", type=Path, default=Path("."), help="ログ保存先")
    args = parser.parse_args()

    cfg = TorqueRampConfig()
    cfg.joint_id = args.joint
    cfg.torque_max = args.max_torque
    cfg.torque_step = args.step
    cfg.strain_threshold = args.strain_limit
    cfg.output_dir = args.log_dir
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    test = TorqueRampTest(FakeController(), FakeSensors(), cfg)
    test.run()

if __name__ == "__main__":
    main()