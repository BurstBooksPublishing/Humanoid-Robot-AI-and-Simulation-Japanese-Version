#!/usr/bin/env python3
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Final, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray


@dataclass
class ImuSample:
    stamp: float
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


class ImuDriver:
    """ハードウェア依存を隠蔽したIMUドライバ"""
    def read(self) -> ImuSample:
        # TODO: 実機ではI2C/SPI経由で読み出す
        raise NotImplementedError


class TorqueDriver:
    """モータードライバへの非ブロッキング送信インターフェース"""
    def send(self, tau: float) -> None:
        # TODO: 実機ではCAN/USB 送信
        raise NotImplementedError


class ImuBuffer:
    """最新サンプルをロックフリーで取得するリングバッファ"""
    def __init__(self, maxlen: int = 256) -> None:
        self._buf: Deque[ImuSample] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, sample: ImuSample) -> None:
        with self._lock:
            self._buf.append(sample)

    def latest(self) -> Optional[ImuSample]:
        with self._lock:
            return self._buf[-1] if self._buf else None


class BalanceController(Node):
    """倒立振子PD制御ノード"""
    # 制御パラメータ
    Ts: Final[float] = 0.002            # 500 Hz
    ALPHA: Final[float] = 0.98          # 相補フィルタ係数
    KP: Final[float] = 12.0
    KD: Final[float] = 0.8

    def __init__(self) -> None:
        super().__init__('balance_controller')
        self._imu_buf = ImuBuffer()
        self._theta = 0.0
        self._torque = TorqueDriver()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self._pub = self.create_publisher(Float32MultiArray, 'joint_command', qos)

        # スレッド起動
        self._imu_thread = threading.Thread(target=self._imu_reader, daemon=True)
        self._ctl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._imu_thread.start()
        self._ctl_thread.start()

    def _imu_reader(self) -> None:
        """IMU取得スレッド（1 kHz）"""
        imu = ImuDriver()
        while rclpy.ok():
            raw = imu.read()
            stamp = time.time()
            self._imu_buf.append(
                ImuSample(
                    stamp=stamp,
                    acc_x=raw.acc_x,
                    acc_y=raw.acc_y,
                    acc_z=raw.acc_z,
                    gyro_x=raw.gyro_x,
                    gyro_y=raw.gyro_y,
                    gyro_z=raw.gyro_z,
                )
            )
            time.sleep(0.001)

    def _control_loop(self) -> None:
        """500 Hz PD制御ループ"""
        next_time = time.time()
        while rclpy.ok():
            sample = self._imu_buf.latest()
            if sample is None:
                continue

            # 相補フィルタで姿勢推定
            acc_angle = math.atan2(sample.acc_y, sample.acc_z)
            self._theta = (
                self.ALPHA * (self._theta + sample.gyro_z * self.Ts)
                + (1.0 - self.ALPHA) * acc_angle
            )

            # PD制御
            error = -self._theta
            u = self.KP * error - self.KD * sample.gyro_z
            self._publish_torque(u)

            # 周期維持
            next_time += self.Ts
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.time()  # デッドラインミス再同期

    def _publish_torque(self, tau: float) -> None:
        msg = Float32MultiArray()
        msg.data = [tau]
        self._pub.publish(msg)
        self._torque.send(tau)


def main(args=None):
    rclpy.init(args=args)
    node = BalanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()