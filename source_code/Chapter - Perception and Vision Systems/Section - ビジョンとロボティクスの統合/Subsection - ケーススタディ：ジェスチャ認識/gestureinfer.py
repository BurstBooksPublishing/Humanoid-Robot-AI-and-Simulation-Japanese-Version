import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import mediapipe as mp


class GestureNode(Node):
    def __init__(self):
        super().__init__('gesture_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', 'gesture_lstm.pt'),
                ('window_size', 30),
                ('conf_thresh', 0.8),
                ('hysteresis', 3),
                ('camera_id', 0),
            ]
        )

        self.model = torch.jit.load(self.get_parameter('model_path').value)
        self.model.eval()

        self.W = self.get_parameter('window_size').value
        self.tau = self.get_parameter('conf_thresh').value
        self.k_req = self.get_parameter('hysteresis').value

        self.window = []
        self.consensus = []

        self.gest_pub = self.create_publisher(Int32, 'gesture_id', 10)

        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(self.get_parameter('camera_id').value)
        if not self.cap.isOpened():
            self.get_logger().error("カメラが開けません")
            rclpy.shutdown()

        timer_period = 0.033  # 30 FPS
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def extract_keypoints(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(img_rgb)
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            return pts
        return None

    def normalize_keypoints(self, pts):
        # 手首を原点に移動＋スケール正規化
        pts = pts - pts[0]
        scale = np.linalg.norm(pts[9] - pts[0])
        if scale > 0:
            pts /= scale
        return pts.flatten()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        keypoints = self.extract_keypoints(frame)
        if keypoints is None:
            self.window.clear()
            return

        vec = self.normalize_keypoints(keypoints)
        self.window.append(vec)
        if len(self.window) > self.W:
            self.window.pop(0)

        if len(self.window) == self.W:
            seq = torch.tensor(np.stack(self.window)).unsqueeze(0).float()
            with torch.no_grad():
                logits = self.model(seq)
                probs = torch.softmax(logits[0], dim=0).numpy()
            label = int(np.argmax(probs))
            conf = float(np.max(probs))

            self.consensus.append((label, conf))
            if len(self.consensus) > self.k_req:
                self.consensus.pop(0)

            # 連続フレームで閾値超えで確定
            if all(l == label and c >= self.tau for l, c in self.consensus):
                msg = Int32()
                msg.data = label
                self.gest_pub.publish(msg)
                self.consensus.clear()


def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()