= []
        self.lock = threading.Lock()

        self.motion_thresh = 0.02
        self.p_th = 0.85
        self.alpha = 0.3
        self.smoothed: Dict[str, float] = {}

        self.timer = self.create_timer(0.1, self.process_buffer)  # 10Hz処理

    def cb_skeleton(self, msg: String) -> None:
        pose = self.deserialize(msg.data)
        with self.lock:
            self.buffer.append(pose)

    def process_buffer(self) -> None:
        with self.lock:
            if len(self.buffer) < 2:
                return
            buf = self.buffer.copy()

        if self.motion_energy(buf) > self.motion_thresh:
            pred, conf = self.classify(buf)
            self.smoothed[pred] = self.alpha * conf + \
                (1 - self.alpha) * self.smoothed.get(pred, 0.0)
            if self.smoothed[pred] >= self.p_th:
                out = String()
                out.data = pred
                self.pub.publish(out)
                with self.lock:
                    self.buffer.clear()

    def motion_energy(self, buf: List[np.ndarray]) -> float:
        if len(buf) < 2:
            return 0.0
        diffs = np.diff(np.array(buf), axis=0)
        return float(np.mean(np.sum(diffs ** 2, axis=1)))

    def classify(self, buf: List[np.ndarray]) -> Tuple[str, float]:
        # 実際のモデル推論に置き換える
        return ("wave", 0.9)

    def deserialize(self, s: str) -> np.ndarray:
        return np.fromstring(s, sep=',')


def main(args=None):
    rclpy.init(args=args)
    node = GestureRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()