#!/usr/bin/env python3
import os
import threading
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 意図数とラベル
NUM_INTENTS = 5
INTENT_LABELS = ["goto", "pick", "stop", "status", "fallback"]

# GPU/CPU 自動選択
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IntentRecognizer(Node):
    def __init__(self) -> None:
        super().__init__("intent_recognizer")

        # モデル読み込み（一度だけ）
        model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=NUM_INTENTS
        ).to(DEVICE).eval()

        # 信念と遷移行列の初期化
        self.belief: np.ndarray = np.ones(NUM_INTENTS) / NUM_INTENTS
        self.trans_matrix: np.ndarray = np.eye(NUM_INTENTS)  # 単位行列で簡略化

        # ロック付き共有変数
        self._lock = threading.Lock()

        # ROS 2 購読・配信
        self.sub = self.create_subscription(
            String, "/asr_hypotheses", self._asr_cb, 10
        )
        self.pub = self.create_publisher(Float32MultiArray, "/intent_belief", 10)

    def _intent_probs_from_text(self, text: str) -> np.ndarray:
        """1文から意図確率を返す"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128
            ).to(DEVICE)
            logits = self.model(**inputs).logits[0].cpu().numpy()
        # ソフトマックス
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()

    def _aggregate_intent(self, hyps: List[Tuple[str, float]]) -> np.ndarray:
        """ASR仮説リストから統合意図分布を計算"""
        agg = np.zeros(NUM_INTENTS)
        for h_text, p_h in hyps:
            agg += self._intent_probs_from_text(h_text) * p_h
        return agg / (agg.sum() or 1.0)

    def _belief_update(self, obs_probs: np.ndarray) -> None:
        """POMDP信念更新（予測→観測更新→正規化）"""
        with self._lock:
            pred = self.trans_matrix.T @ self.belief
            new_b = obs_probs * pred
            self.belief = new_b / (new_b.sum() or 1.0)

    def _asr_cb(self, msg: String) -> None:
        """"/asr_hypotheses" コールバック"""
        # 簡易フォーマット: "text1,score1;text2,score2;..."
        try:
            hyps = [
                (part.split(",")[0], float(part.split(",")[1]))
                for part in msg.data.split(";")
                if part
            ]
        except (IndexError, ValueError):
            self.get_logger().warn("Invalid ASR format")
            return

        obs = self._aggregate_intent(hyps)
        self._belief_update(obs)

        # 配信
        out = Float32MultiArray()
        out.data = self.belief.tolist()
        self.pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = IntentRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()