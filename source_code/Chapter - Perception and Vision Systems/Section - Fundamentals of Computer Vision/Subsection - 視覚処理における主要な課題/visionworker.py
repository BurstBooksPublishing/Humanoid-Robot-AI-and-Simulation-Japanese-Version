import cv2
import numpy as np
import threading
import time
from typing import Callable, Tuple, Optional

Frame = np.ndarray
Depth = np.ndarray
Confidence = np.ndarray
Timestamp = float
PublishFn = Callable[[Frame, Depth, Confidence, Timestamp], None]

class VisionWorker(threading.Thread):
    def __init__(
        self,
        frame_source: "FrameSource",  # 依存性注入でテスタブルに
        publish_fn: PublishFn,
        fps: float = 30.0,
        name: str = "VisionWorker",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._src = frame_source
        self._publish = publish_fn
        self._running = threading.Event()
        self._running.set()
        self._period = 1.0 / fps
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[Frame, Depth, Timestamp]] = None

    def run(self) -> None:
        while self._running.is_set():
            t0 = time.perf_counter()
            color, depth, ts = self._src.read()
            if color is None or depth is None:
                continue

            # ガンマ補正＋CLAHEで露出補正（HDR対策）
            lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            color_out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

            # 深度信頼度マスク生成
            valid = (depth > 0.1) & (depth < 5.0)
            median = cv2.medianBlur(depth, 5)
            conf = np.abs(depth - median) < 0.05
            depth_conf = valid & conf

            self._publish(color_out, depth, depth_conf, ts)

            # FPS制御
            elapsed = time.perf_counter() - t0
            sleep = max(0.0, self._period - elapsed)
            time.sleep(sleep)

    def stop(self) -> None:
        self._running.clear()
        self.join(timeout=1.0)