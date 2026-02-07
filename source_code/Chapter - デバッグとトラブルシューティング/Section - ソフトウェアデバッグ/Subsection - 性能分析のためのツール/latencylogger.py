import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image  # 実際のメッセージ型
import numpy as np
import csv
import os
import atexit
from typing import List

class LatencyLogger(Node):
    def __init__(self) -> None:
        super().__init__('latency_logger')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub = self.create_subscription(
            Image,
            '/camera/image_stamped',
            self.cb,
            qos
        )
        self.latencies: List[float] = []
        self.csv_path = 'latency_stats.csv'
        # 既存ファイルを上書きしないように初回のみヘッダ書き込み
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(['count', 'mean', 'p95', 'p99'])
        atexit.register(self._write_stats)  # 終了時にも出力

    def cb(self, msg: Image) -> None:
        now = self.get_clock().now().to_msg()
        send_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        recv_ts = now.sec + now.nanosec * 1e-9
        lat = recv_ts - send_ts
        self.latencies.append(lat)
        if len(self.latencies) % 1000 == 0:
            self._write_stats()

    def _write_stats(self) -> None:
        if not self.latencies:
            return
        a = np.array(self.latencies)
        mean = float(a.mean())
        p95 = float(np.percentile(a, 95))
        p99 = float(np.percentile(a, 99))
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([len(a), mean, p95, p99])

def main(args=None) -> None:
    rclpy.init(args=args)
    node = LatencyLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()