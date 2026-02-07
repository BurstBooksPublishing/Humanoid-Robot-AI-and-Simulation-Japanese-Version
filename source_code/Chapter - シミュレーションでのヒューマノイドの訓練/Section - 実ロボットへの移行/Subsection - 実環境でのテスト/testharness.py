#!/usr/bin/env python3
import os
import signal
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool
from sensor_msgs.msg import Imu

# QoS設定：センサデータ用
IMU_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
)

class TestHarness(Node):
    def __init__(self) -> None:
        super().__init__('test_harness')
        self._shutdown_evt = threading.Event()

        # 緊急停止監視
        self.emergency_sub = self.create_subscription(
            Bool, '/safety/emergency', self._emergency_cb, 10)

        # IMU受信
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self._imu_cb, IMU_QOS)

        # ログディレクトリ作成
        log_dir = Path('/var/log/humanoid_tests')
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f'session_{int(time.time())}.log'
        self.log_file = self.log_path.open('a', buffering=1)  # line buffering

        self.emergency = False
        self._lock = threading.Lock()

        # シグナルハンドラ登録
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        self.get_logger().info('Shutdown signal received')
        self._shutdown_evt.set()

    def _emergency_cb(self, msg: Bool) -> None:
        with self._lock:
            self.emergency = msg.data
        if self.emergency:
            self.get_logger().warn('Emergency stop triggered')

    def _imu_cb(self, msg: Imu) -> None:
        with self._lock:
            if self.emergency:
                return
        # タイムスタンプと加速度をログ
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.log_file.write(
            f'{ts},{msg.linear_acceleration.x},'
            f'{msg.linear_acceleration.y},{msg.linear_acceleration.z}\n')

    def run_trial(self, task_fn: Callable[[], None], timeout_s: float = 30.0) -> bool:
        if self._shutdown_evt.is_set():
            return False

        start = self.get_clock().now()
        task_fn()

        while (self.get_clock().now() - start).nanoseconds * 1e-9 < timeout_s:
            if self._shutdown_evt.wait(0.01):
                return False
            with self._lock:
                if self.emergency:
                    self.get_logger().error('Abort: emergency stop')
                    return False
        return True

    def destroy_node(self) -> None:
        self.log_file.close()
        super().destroy_node()

def main(args=None) -> None:
    rclpy.init(args=args)
    node = TestHarness()
    try:
        # 実際のtask_fnは呼び出し側で定義
        # node.run_trial(task_fn)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()