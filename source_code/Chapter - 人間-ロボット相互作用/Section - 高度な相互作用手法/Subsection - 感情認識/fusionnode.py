import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32MultiArray

# --- モデル定義 ---
class FusionHead(nn.Module):
    def __init__(self, emb_dim: int = 512, hidden: int = 128, n_classes: int = 6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)

# --- 推論ラッパー ---
class MultimodalInference:
    def __init__(self,
                 face_path: str,
                 speech_path: str,
                 fusion_path: str,
                 device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # TorchScriptモデルを読み込み
        self.face_embedder = torch.jit.load(face_path, map_location=self.device).eval()
        self.speech_embedder = torch.jit.load(speech_path, map_location=self.device).eval()

        # FusionHeadを構築し重みを読み込み
        self.fusion = FusionHead(emb_dim=512, hidden=128, n_classes=6).to(self.device)
        self.fusion.load_state_dict(torch.load(fusion_path, map_location=self.device))
        self.fusion.eval()

    @torch.no_grad()
    def infer(self, face_frame: torch.Tensor, audio_buffer: torch.Tensor) -> Tuple[int, np.ndarray]:
        # 入力をデバイスへ移動
        face_frame = face_frame.to(self.device, non_blocking=True)
        audio_buffer = audio_buffer.to(self.device, non_blocking=True)

        zf = self.face_embedder(face_frame)
        zs = self.speech_embedder(audio_buffer)
        z = torch.cat([zf, zs], dim=-1)  # 結合
        logits = self.fusion(z)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        label = int(np.argmax(probs))
        return label, probs

# --- ROS 2ノード ---
class MultimodalNode(Node):
    def __init__(self):
        super().__init__("multimodal_inference")
        self.declare_parameter("face_model_path", "face_embedder.pt")
        self.declare_parameter("speech_model_path", "speech_embedder.pt")
        self.declare_parameter("fusion_model_path", "fusion_head.pt")

        face_path = self.get_parameter("face_model_path").value
        speech_path = self.get_parameter("speech_model_path").value
        fusion_path = self.get_parameter("fusion_model_path").value

        self.inference = MultimodalInference(face_path, speech_path, fusion_path)

        # サブスクライバ／パブリッシャ
        self.face_sub = self.create_subscription(
            Float32MultiArray, "/face_tensor", self.face_cb, 1)
        self.audio_sub = self.create_subscription(
            Float32MultiArray, "/audio_tensor", self.audio_cb, 1)
        self.label_pub = self.create_publisher(Int32, "/predicted_label", 1)
        self.prob_pub = self.create_publisher(Float32MultiArray, "/predicted_probs", 1)

        self.face_tensor = None
        self.audio_tensor = None

    def face_cb(self, msg: Float32MultiArray):
        self.face_tensor = torch.tensor(msg.data, dtype=torch.float32).unsqueeze(0)
        self.try_infer()

    def audio_cb(self, msg: Float32MultiArray):
        self.audio_tensor = torch.tensor(msg.data, dtype=torch.float32).unsqueeze(0)
        self.try_infer()

    def try_infer(self):
        if self.face_tensor is None or self.audio_tensor is None:
            return
        label, probs = self.inference.infer(self.face_tensor, self.audio_tensor)
        self.label_pub.publish(Int32(data=label))
        self.prob_pub.publish(Float32MultiArray(data=probs.flatten().tolist()))

def main(args=None):
    rclpy.init(args=args)
    node = MultimodalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()