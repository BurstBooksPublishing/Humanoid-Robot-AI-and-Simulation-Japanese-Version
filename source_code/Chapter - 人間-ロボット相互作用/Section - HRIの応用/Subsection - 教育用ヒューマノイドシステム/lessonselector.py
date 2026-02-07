import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from typing import Dict, Any, Optional
import json
import numpy as np
from ament_index_python import get_package_share_directory
import os

from tutorial_interfaces.msg import SensorObs, LessonCmd   # プロジェクト固有のメッセージ


class LessonSelector(Node):
    """レッスン選択ノード：観測に基づくベイズ更新＋期待効用最大化"""

    def __init__(self) -> None:
        super().__init__('lesson_selector')

        # パラメータ宣言
        self.declare_parameter('likelihood_path', '')
        self.declare_parameter('transition_path', '')
        self.declare_parameter('reward_path', '')
        self.add_on_set_parameters_callback(self._param_cb)

        # QoS：センサ観測はベストエフォート、レッスン命令は信頼性重視
        sub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        pub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self._sub = self.create_subscription(
            SensorObs, 'sensor_obs', self._obs_cb, sub_qos)
        self._pub = self.create_publisher(LessonCmd, 'lesson_cmd', pub_qos)

        self._states = ['novice', 'intermediate', 'advanced']
        self._belief: Dict[str, float] = {s: 1.0 / len(self._states) for s in self._states}

        self._likelihood: Dict[str, Dict[str, float]] = {}
        self._transition: Dict[str, Dict[str, float]] = {}
        self._reward: Dict[str, Dict[str, float]] = {}

        self._load_models()  # ファイル読み込み

    def _param_cb(self, params) -> SetParametersResult:
        """パラメータ変更時に再読み込み"""
        for p in params:
            if p.name.endswith('_path'):
                self._load_models()
                break
        return SetParametersResult(successful=True)

    def _load_models(self) -> None:
        """JSONファイルから確率・報酬モデルを読み込む"""
        pkg_dir = get_package_share_directory('lesson_selector')
        try:
            with open(os.path.join(pkg_dir, self.get_parameter('likelihood_path').value)) as f:
                self._likelihood = json.load(f)
            with open(os.path.join(pkg_dir, self.get_parameter('transition_path').value)) as f:
                self._transition = json.load(f)
            with open(os.path.join(pkg_dir, self.get_parameter('reward_path').value)) as f:
                self._reward = json.load(f)
            self.get_logger().info("モデル読み込み完了")
        except Exception as e:
            self.get_logger().error(f"モデル読み込み失敗: {e}")

    def _obs_cb(self, msg: SensorObs) -> None:
        """観測到着時：ベイズ更新→行動選択→publish"""
        obs = msg.observation  # string フィールドと仮定

        # 予測ステップ
        prior = {
            s: sum(self._transition[prev][s] * self._belief[prev] for prev in self._states)
            for s in self._states
        }

        # 更新ステップ
        posterior = {
            s: self._likelihood[s].get(obs, 1e-6) * prior[s]
            for s in self._states
        }
        norm = sum(posterior.values())
        if norm > 0:
            self._belief = {s: posterior[s] / norm for s in self._states}
        else:
            self.get_logger().warn("正規化定数ゼロ：信念更新スキップ")

        # 期待効用最大化
        best_action = max(
            self._reward.keys(),
            key=lambda a: sum(self._belief[s] * self._reward[a][s] for s in self._states)
        )

        # メッセージ生成 & 送信
        cmd = LessonCmd()
        cmd.action = best_action
        self._pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LessonSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()