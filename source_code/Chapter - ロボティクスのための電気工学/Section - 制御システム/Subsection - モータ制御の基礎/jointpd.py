#!/usr/bin/env python3
import time
import numpy as np
import threading
from typing import Tuple, Optional

# 制御パラメータ
Kp: float = 150.0
Kd: float = 5.0
MAX_TAU: float = 25.0  # Nm
SAMPLE_DT: float = 0.002  # 2 ms

# グローバル停止フラグ
running: threading.Event = threading.Event()
running.set()


def gravity_comp(theta: float) -> float:
    """リンクの近似重力補償トルク"""
    return 9.81 * 0.5 * 1.0 * np.sin(theta)


def read_sensors() -> Tuple[float, float]:
    """エンコーダとIMUから角度・角速度を取得（ダミー実装）"""
    # TODO: 実際のハードウェアインターフェースに置き換え
    return 0.0, 0.0


def compute_trajectory(t: float) -> Tuple[float, float, float]:
    """時刻tにおける目標角度・角速度・フィードフォワードトルク（ダミー）"""
    # TODO: 軌道生成器に置き換え
    return 0.0, 0.0, 0.0


def send_torque(tau: float) -> None:
    """ドライバへトルク指令送信（ダミー）"""
    # TODO: 実際のCAN/EtherCAT等に置き換え
    pass


def control_loop() -> None:
    """2 ms周期で実行されるリアルタイム制御ループ"""
    prev_error: float = 0.0
    t_start: float = time.perf_counter()

    while running.is_set():
        t0: float = time.perf_counter()

        # 時刻取得
        t: float = t0 - t_start

        # センサ読み出し
        theta, theta_dot = read_sensors()

        # 目標値生成
        theta_des, theta_dot_des, tau_ff = compute_trajectory(t)

        # PD+FF+重力補償
        error: float = theta_des - theta
        derror: float = (error - prev_error) / SAMPLE_DT
        tau_cmd: float = (
            Kp * error
            + Kd * (theta_dot_des - theta_dot)
            + tau_ff
            + gravity_comp(theta)
        )

        # 安全クランプ
        tau_cmd = np.clip(tau_cmd, -MAX_TAU, MAX_TAU)

        # 指令送信
        send_torque(tau_cmd)

        prev_error = error

        # 周期同期
        elapsed: float = time.perf_counter() - t0
        time.sleep(max(0.0, SAMPLE_DT - elapsed))


def main() -> None:
    """メインエントリ：スレッド起動＆安全停止"""
    try:
        control_loop()
    except KeyboardInterrupt:
        running.clear()
        # トルクをゼロにして終了
        send_torque(0.0)


if __name__ == "__main__":
    main()