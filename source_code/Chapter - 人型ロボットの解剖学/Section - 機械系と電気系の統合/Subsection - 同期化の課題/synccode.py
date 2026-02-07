import time
from typing import Optional, Dict, Tuple, Any
import numpy as np
from collections import deque
import threading

class SensorBuffer:
    """
    時系列データを保持し、線形補間で任意時刻の値を返すスレッドセーフなバッファ。
    """
    def __init__(self, maxlen: int = 200) -> None:
        self._buf: deque[Tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, t: float, value: np.ndarray) -> None:
        with self._lock:
            self._buf.append((t, value.copy()))

    def interpolate(self, t_query: float) -> Optional[np.ndarray]:
        with self._lock:
            if len(self._buf) < 2:
                return None
            times = np.fromiter((t for t, _ in self._buf), dtype=float)
            if t_query < times[0] or t_query > times[-1]:
                return None
            idx = int(np.searchsorted(times, t_query))
            t0, t1 = times[idx - 1], times[idx]
            v0, v1 = self._buf[idx - 1][1], self._buf[idx][1]
            alpha = (t_query - t0) / (t1 - t0)
            return (1.0 - alpha) * v0 + alpha * v1


class SyncedSensorManager:
    """
    IMU・エンコーダのバッファを管理し、指定時刻で同期した観測を返す。
    """
    def __init__(self, maxlen: int = 200) -> None:
        self.imu_buf = SensorBuffer(maxlen=maxlen)
        self.enc_buf = SensorBuffer(maxlen=maxlen)

    def on_imu(self, t_hw: float, acc_gyro: np.ndarray) -> None:
        self.imu_buf.append(t_hw, acc_gyro)

    def on_encoder(self, t_hw: float, q: np.ndarray) -> None:
        self.enc_buf.append(t_hw, q)

    def get_synced(self, t_ref: float) -> Optional[Dict[str, Any]]:
        imu = self.imu_buf.interpolate(t_ref)
        enc = self.enc_buf.interpolate(t_ref)
        if imu is None or enc is None:
            return None
        return {"t": t_ref, "imu": imu, "enc": enc}