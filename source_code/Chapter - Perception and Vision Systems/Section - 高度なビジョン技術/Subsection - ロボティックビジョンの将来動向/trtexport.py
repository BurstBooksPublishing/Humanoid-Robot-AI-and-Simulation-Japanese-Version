import os
import time
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.quantization as quant
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 定数
MODEL_HUB = "pytorch/vision:v0.10.0"
MODEL_NAME = "deeplabv3_resnet50"
ONNX_PATH = Path("seg_model.onnx")
ENGINE_PATH = Path("seg_model.trt")
CALIB_FRAMES_DIR = Path("calib_frames")
INPUT_SHAPE = (1, 3, 480, 640)  # humanoid head camera
BATCH_SIZE = 1
MAX_CALIB_BATCH = 10
TRT_MAX_WORKSPACE = 1 << 30  # 1GB
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class CalibDataset(torch.utils.data.Dataset):
    """キャリブレーション用の簡易データセット"""
    def __init__(self, root: Path, max_samples: int = MAX_CALIB_BATCH):
        self.imgs = sorted(root.glob("*.jpg"))[:max_samples]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB").resize((640, 480))
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

def quantize_model(model: torch.nn.Module, save_path: Path) -> torch.nn.Module:
    """与えられたモデルを静的量子化し保存"""
    model.eval()
    model.qconfig = quant.get_default_qconfig("fbgemm")
    quant.prepare(model, inplace=True)

    # 校正
    dataset = CalibDataset(CALIB_FRAMES_DIR)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for x in loader:
            _ = model(x)

    quant.convert(model, inplace=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"量子化モデルを保存: {save_path}")
    return model

def export_onnx(model: torch.nn.Module, path: Path) -> None:
    """PyTorchモデルをONNX形式でエクスポート"""
    dummy = torch.randn(INPUT_SHAPE)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    onnx.checker.check_model(str(path))
    logger.info(f"ONNXモデルを保存: {path}")

def build_engine(onnx_path: Path, engine_path: Path) -> trt.ICudaEngine:
    """ONNXからTensorRTエンジンをビルド"""
    if engine_path.exists():
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = TRT_MAX_WORKSPACE
    config.set_flag(trt.BuilderFlag.FP16)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    engine = builder.build_engine(network, config)
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    logger.info(f"TensorRTエンジンを保存: {engine_path}")
    return engine

class TRTInference:
    """TensorRT推論ラッパー"""
    def __init__(self, engine: trt.ICudaEngine):
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()

        # バッファ確保
        self.host_in = cuda.pagelocked_empty(
            trt.volume(engine.get_binding_shape(0)), dtype=np.float32
        )
        self.host_out = cuda.pagelocked_empty(
            trt.volume(engine.get_binding_shape(1)), dtype=np.float32
        )
        self.dev_in = cuda.mem_alloc(self.host_in.nbytes)
        self.dev_out = cuda.mem_alloc(self.host_out.nbytes)

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """1フレーム推論"""
        np.copyto(self.host_in, frame.ravel())
        cuda.memcpy_htod_async(self.dev_in, self.host_in, self.stream)
        self.context.execute_async_v2(
            bindings=[int(self.dev_in), int(self.dev_out)], stream_handle=self.stream.handle
        )
        cuda.memcpy_dtoh_async(self.host_out, self.dev_out, self.stream)
        self.stream.synchronize()
        return self.host_out.reshape(self.engine.get_binding_shape(1))

def main():
    # モデル取得・量子化
    model = torch.hub.load(MODEL_HUB, MODEL_NAME, pretrained=True)
    quantize_model(model, Path("seg_model_q.pth"))

    # ONNXエクスポート
    export_onnx(model, ONNX_PATH)

    # TensorRTエンジン構築
    engine = build_engine(ONNX_PATH, ENGINE_PATH)

    # 推論ループ
    trt_inf = TRTInference(engine)
    cap = cv2.VideoCapture(0)  # カメラ例
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        input_blob = np.expand_dims(frame.transpose(2, 0, 1), 0).astype(np.float32) / 255.0
        seg = trt_inf.infer(input_blob)
        # 後処理例: segを可視化
        cv2.imshow("seg", seg[0])
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()