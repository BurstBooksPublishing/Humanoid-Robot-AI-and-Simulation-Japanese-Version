#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from typing import List, Tuple

# 仮想APIのROS 2ラッパー
class KneeDriver(Node):
    def __init__(self) -> None:
        super().__init__('knee_driver')
        self._cmd_pub = self.create_publisher(Float64, '/right_knee/command', 1)
        self._state_sub = self.create_subscription(
            JointState, '/right_knee/state', self._state_cb, 1)
        self._latest_state: JointState = JointState()

    def _state_cb(self, msg: JointState) -> None:
        self._latest_state = msg

    def command_torque(self, tau: float) -> None:
        self._cmd_pub.publish(Float64(data=tau))

    def read_state(self) -> Tuple[float, float, float]:
        # pos[rad], vel[rad/s], current[A]
        return (self._latest_state.position[0],
                self._latest_state.velocity[0],
                self._latest_state.effort[0])

# 定数
K_T: float = 0.08          # Nm/A トルク定数
SAMPLE_HZ: int = 1000
STEPS: List[float] = [0.0, 2.0, -2.0, 0.0]  # Nm
STEP_DURATION: float = 2.0  # s

def main() -> None:
    rclpy.init()
    driver = KneeDriver()

    log: List[Tuple[float, float, float, float, float, float]] = []

    for tau_cmd in STEPS:
        driver.command_torque(tau_cmd)
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < STEP_DURATION:
            rclpy.spin_once(driver, timeout_sec=0.0)
            pos, vel, cur = driver.read_state()
            est_tau = K_T * cur
            log.append((time.perf_counter(), tau_cmd, pos, vel, cur, est_tau))
            time.sleep(1.0 / SAMPLE_HZ)

    driver.command_torque(0.0)  # 安全停止
    np.save('torque_step_log.npy', np.array(log, dtype=np.float64))
    rclpy.shutdown()

if __name__ == '__main__':
    main()