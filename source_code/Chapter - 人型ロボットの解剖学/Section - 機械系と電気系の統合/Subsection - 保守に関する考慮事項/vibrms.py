#!/usr/bin/env python3
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Final

import board
import busio
import adafruit_adxl34x  # I2C加速度センサ例

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: Final = logging.getLogger("vibration_monitor")

# 定数
WINDOW: Final = 256          # ローリングウィンドウ長
THRESHOLD_G: Final = 0.75    # RMS閾値[g]
SAMPLE_HZ: Final = 100       # サンプリング周波数[Hz]
SLEEP_SEC: Final = 1.0 / SAMPLE_HZ

# センサ初期化
i2c = busio.I2C(board.SCL, board.SDA)
accel = adafruit_adxl34x.ADXL345(i2c)
accel.range = adafruit_adxl34x.Range.RANGE_16_G  # ±16G


@dataclass(slots=True, frozen=True)
class VibrationEvent:
    rms: float
    timestamp: float = time.time()


class VibrationMonitor:
    def __init__(self, window: int = WINDOW, threshold_g: float = THRESHOLD_G) -> None:
        self._buf: Deque[float] = deque(maxlen=window)
        self._threshold: Final = threshold_g

    def _read_accel_magnitude(self) -> float:
        """ADXL345から加速度大きさ[g]を取得"""
        x, y, z = accel.acceleration
        return math.sqrt(x * x + y * y + z * z) / 9.80665  # m/s² → g

    def _rolling_rms(self) -> float:
        """バッファ内のRMS[g]を計算"""
        n = len(self._buf)
        if n == 0:
            return 0.0
        return math.sqrt(sum(v * v for v in self._buf) / n)

    def _handle_alert(self, rms: float) -> None:
        """閾値超過時のログ出力と保守依頼"""
        logger.warning("VIBRATION_ALERT rms=%.3f g", rms)
        # TODO: 保守システムへのREST呼出等を実装
        # trigger_maintenance_workorder()

    def run(self) -> None:
        """メインループ"""
        logger.info("Vibration monitor started")
        while True:
            self._buf.append(self._read_accel_magnitude())
            if len(self._buf) == self._buf.maxlen:
                rms = self._rolling_rms()
                if rms > self._threshold:
                    self._handle_alert(rms)
            time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    try:
        VibrationMonitor().run()
    except KeyboardInterrupt:
        logger.info("Shutting down")