import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32, String
import threading
import yaml
import os
from ament_index_python.packages import get_package_share_directory


class AdaptiveLessonNode(Node):
    def __init__(self):
        super().__init__('adaptive_lesson')

        # パラメータ宣言＆デフォルト値
        self.declare_parameter('difficulty_init', 0.5)
        self.declare_parameter('alpha', 0.3)
        self.declare_parameter('beta', 0.2)
        self.declare_parameter('target_success', 0.8)
        self.declare_parameter('target_engagement', 0.6)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('config_file', '')

        # パラメータ読み込み
        self.d = self.get_parameter('difficulty_init').value
        self.alpha = self.get_parameter('alpha').value
        self.beta = self.get_parameter('beta').value
        self.target_success = self.get_parameter('target_success').value
        self.target_engagement = self.get_parameter('target_engagement').value
        rate_hz = self.get_parameter('rate_hz').value

        # YAML設定ファイルがあれば上書き
        config_path = self.get_parameter('config_file').value
        if config_path and os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                self.d = float(cfg.get('difficulty_init', self.d))
                self.alpha = float(cfg.get('alpha', self.alpha))
                self.beta = float(cfg.get('beta', self.beta))
                self.target_success = float(cfg.get('target_success', self.target_success))
                self.target_engagement = float(cfg.get('target_engagement', self.target_engagement))

        # 購読／配信
        self.create_subscription(Float32, '/engagement', self.eng_cb, 10)
        self.create_subscription(Float32, '/success', self.succ_cb, 10)
        self.pub = self.create_publisher(String, '/behavior_cmd', 10)

        # 状態変数
        self.last_eng = 0.0
        self.last_succ = 0.0
        self.lock = threading.Lock()

        # パラメータ変更コールバック
        self.add_on_set_parameters_callback(self.param_cb)

        # タイマー駆動で更新（受信タイミングに依存しない）
        self.timer = self.create_timer(1.0 / rate_hz, self.update_and_issue)

    def eng_cb(self, msg):
        with self.lock:
            self.last_eng = msg.data

    def succ_cb(self, msg):
        with self.lock:
            self.last_succ = msg.data

    def param_cb(self, params):
        # 実行中のパラメータ更新を許容
        for p in params:
            if p.name == 'alpha':
                self.alpha = p.value
            elif p.name == 'beta':
                self.beta = p.value
            elif p.name == 'target_success':
                self.target_success = p.value
            elif p.name == 'target_engagement':
                self.target_engagement = p.value
        return SetParametersResult(successful=True)

    def update_and_issue(self):
        with self.lock:
            eng = self.last_eng
            succ = self.last_succ

        # 難易度更新（式2）
        delta = self.alpha * (succ - self.target_success) + self.beta * (eng - self.target_engagement)
        self.d = max(0.0, min(1.0, self.d + delta))

        cmd = self.select_behavior(succ, eng)
        self.pub.publish(String(data=cmd))

    def select_behavior(self, succ, eng):
        # 状態に応じた振る舞い選択
        if succ < 0.5:
            return 'hint'
        if eng < 0.4:
            return 'reengage'
        if self.d < 0.4:
            return 'practice'
        if self.d < 0.8:
            return 'challenge'
        return 'assessment'


def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveLessonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()