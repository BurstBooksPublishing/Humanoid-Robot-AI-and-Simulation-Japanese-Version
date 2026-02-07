import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自動初期化
import threading
import queue
import time
from typing import Tuple, Optional

# TensorRT エンジン読み込み
def load_engine(engine_path: str) -> trt.ICudaEngine:
    with open(engine_path, "rb") as f, trt.Logger() as logger, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# 非同期推論ラッパー
class TRTPredictor:
    def __init__(self, engine_path: str, stream: cuda.Stream):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = stream
        self.bindings = []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.host_in = host_mem
                self.device_in = device_mem
            else:
                self.host_out = host_mem
                self.device_out = device_mem

    def infer(self, img: np.ndarray) -> np.ndarray:
        np.copyto(self.host_in, img.ravel())
        cuda.memcpy_htod_async(self.device_in, self.host_in, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_out, self.device_out, self.stream)
        self.stream.synchronize()
        out_shape = self.engine.get_binding_shape(1)
        return self.host_out.reshape(out_shape)

# 前処理
def preprocess(frame: np.ndarray) -> np.ndarray:
    img = cv2.resize(frame, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return np.transpose(img, (2, 0, 1)).ravel()

# 後処理
def postprocess(logits: np.ndarray, depth: Optional[np.ndarray] = None, min_vol: float = 0.002) -> np.ndarray:
    seg = np.argmax(logits, axis=0).astype(np.uint8)
    if depth is not None:
        # 深度が小さい領域を除外
        mask = depth > 0.1
        seg *= mask
    return seg

# 可視化
def visualize_segmentation(frame: np.ndarray, seg: np.ndarray) -> np.ndarray:
    color = cv2.applyColorMap(seg * 50, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.7, color, 0.3, 0)

# メイン
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("カメラが開けません")
    stream = cuda.Stream()
    predictor = TRTPredictor("model.trt", stream)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inp = preprocess(frame)
        logits = predictor.infer(inp)
        seg = postprocess(logits)
        display = visualize_segmentation(frame, seg)
        cv2.imshow("seg", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()