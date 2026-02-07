#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from audio_msgs.msg import AudioMsg
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import torch
import numpy as np
from threading import Lock
import time

class VoiceCmdNode(Node):
    def __init__(self):
        super().__init__('voice_cmd')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        self.sub_audio = self.create_subscription(
            AudioMsg, '/mics/beamed', self.on_audio, qos)
        self.pub_cmd = self.create_publisher(Twist, '/robot/cmd', qos)
        self.pub_status = self.create_publisher(String, '/voice/status', qos)

        self.lock = Lock()
        self.intent_model = torch.jit.load('/opt/models/intent.pt').eval()
        self.vad = torch.jit.load('/opt/models/vad.pt').eval()
        self.safety = SafetyChecker(self.get_logger())
        self.pending = None
        self.timer = None

    def on_audio(self, msg):
        with self.lock:
            audio = np.frombuffer(msg.data, dtype=np.float32)
            if not self.vad(torch.from_numpy(audio)).item():
                return
            text, conf = self.asr(audio)
            if conf < 0.6:
                text, conf = self.edge_asr(audio)
            action, prob = self.classify(text)
            if prob < 0.7 or not self.safety.check(action):
                self.ask_confirm(action)
                return
            self.publish_cmd(action)

    def asr(self, audio):
        # オンデバイスASR
        with torch.no_grad():
            logits = self.intent_model.encode(audio)
            return logits.argmax(-1).item(), logits.softmax(-1).max().item()

    def edge_asr(self, audio):
        # 暗号化エッジ呼び出し（簡略化）
        return "unknown", 0.0

    def classify(self, text):
        # 意図分類
        with torch.no_grad():
            out = self.intent_model(text)
            prob, idx = out.softmax(-1).max(-1)
            return idx.item(), prob.item()

    def ask_confirm(self, action):
        self.pending = action
        self.timer = self.create_timer(3.0, self.timeout)
        self.pub_status.publish(String(data=f"confirm:{action}"))

    def timeout(self):
        self.pending = None
        self.destroy_timer(self.timer)

    def publish_cmd(self, action):
        twist = Twist()
        twist.linear.x = action.get('vx', 0.0)
        twist.angular.z = action.get('wz', 0.0)
        self.pub_cmd.publish(twist)

class SafetyChecker:
    def __init__(self, logger):
        self.logger = logger

    def check(self, action):
        # 速度制限チェック
        if abs(action.get('vx', 0)) > 1.0 or abs(action.get('wz', 0)) > 1.5:
            self.logger.warn("速度制限超過")
            return False
        return True

def main(args=None):
    rclpy.init(args=args)
    node = VoiceCmdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()