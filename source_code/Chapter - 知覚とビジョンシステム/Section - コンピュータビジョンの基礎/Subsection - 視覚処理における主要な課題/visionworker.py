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
    def __init__(self,
                 frame_source,
                 publish_fn: PublishFn,
                 fps: float = 30.0,
                 timeout: float = 1.0):
        super().__init__(daemon=True)
        self._src = frame_source
        self._publish = publish_fn
        self._running = threading.Event()
        self._running.set()
        self._period = 1.0 / fps
        self._timeout = timeout

    def run(self) -> None:
        while self._running.is_set():
            t0 = time.monotonic()
            color, depth, ts = self._src.read()  # ブロッキング読み出し
            if color is None or depth is None:
                continue

            # ヒストグラム均一化で露出補正
            lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            color_out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

            # 深度信頼度マスク生成
            valid = (depth > 0.1) & (depth < 5.0)
            median = cv2.medianBlur(depth, 5)
            conf = np.abs(depth - median) < 0.05
            depth_conf = valid & conf

            self._publish(color_out, depth, depth_conf, ts)

            # FPS制御
            elapsed = time.monotonic() - t0
            sleep = max(0.0, self._period - elapsed)
            time.sleep(sleep)

    def stop(self) -> None:
        self._running.clear()
        self.join(timeout=self._timeout)