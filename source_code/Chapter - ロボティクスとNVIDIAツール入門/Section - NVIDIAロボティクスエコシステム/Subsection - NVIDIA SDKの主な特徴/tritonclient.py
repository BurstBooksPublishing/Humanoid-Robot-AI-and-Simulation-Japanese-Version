import os
import time
import logging
from typing import List, Tuple

import cv2
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

# ログ設定：INFO以上をコンソールへ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# 環境変数からサーバアドレス取得（未設定時はlocalhost）
TRITON_URL = os.getenv("TRITON_HTTP_URL", "localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "humanoid_percept")
INPUT_SHAPE = (224, 224)
INPUT_NAME = "input__0"
OUTPUT_NAME = "output__0"


class TritonInference:
    """Triton HTTP クライアントの薄いラッパー"""

    def __init__(self, url: str, model: str) -> None:
        self.url = url
        self.model = model
        self.client = httpclient.InferenceServerClient(url=url, verbose=False)
        # サーバ・モデルが準備できているか確認
        if not self.client.is_model_ready(model):
            raise RuntimeError(f"モデル {model} が準備できていません")

    def infer(self, img: np.ndarray) -> np.ndarray:
        """1枚の画像を推論し、結果をnumpy配列で返す"""
        inputs = [httpclient.InferInput(INPUT_NAME, img.shape, "FP32")]
        inputs[0].set_data_from_numpy(img)
        outputs = [httpclient.InferRequestedOutput(OUTPUT_NAME)]

        # 同期推論（低遅延）
        try:
            resp = self.client.infer(
                model_name=self.model, inputs=inputs, outputs=outputs
            )
        except InferenceServerException as e:
            logging.error("推論エラー: %s", e)
            raise
        return resp.as_numpy(OUTPUT_NAME)


def preprocess(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """HWC BGR -> resize -> CHW float32 (0-1) -> 4次元化"""
    resized = cv2.resize(frame, size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0)  # batch次元追加


def main() -> None:
    # カメラ初期化（ROS2カメラノード使用時はcv2.VideoCaptureを置き換え）
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        logging.error("カメラが開けません")
        return

    # 解像度固定（処理高速化・帯域削減）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    triton = TritonInference(TRITON_URL, MODEL_NAME)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("フレーム取得失敗")
                time.sleep(0.01)
                continue

            img = preprocess(frame, INPUT_SHAPE)
            result = triton.infer(img)

            # ここに後処理（分類・検出）を追加
            logging.debug("推論結果 shape=%s", result.shape)

    except KeyboardInterrupt:
        logging.info("停止シグナル受信")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()