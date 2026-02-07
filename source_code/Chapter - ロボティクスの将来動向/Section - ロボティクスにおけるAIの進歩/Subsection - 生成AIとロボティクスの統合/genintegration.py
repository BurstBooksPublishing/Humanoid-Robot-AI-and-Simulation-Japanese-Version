import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from typing import List, Optional
import threading
import time


class GenIntegrator(Node):
    def __init__(self) -> None:
        super().__init__('gen_integrator')

        # QoS: ロボットハードウェア向けに信頼性を確保
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_proposal = self.create_subscription(
            JointState, '/gen/proposal', self.cb_prop, qos)
        self.sub_conf = self.create_subscription(
            Float32, '/gen/conf', self.cb_conf, qos)
        self.pub_safe = self.create_publisher(
            JointState, '/robot/joint_cmd', qos)

        # スレッドセーフな共有変数
        self._lock = threading.Lock()
        self.nominal: Optional[List[float]] = None
        self.last_conf: float = 0.0

        # パラメータ読み込み
        self.declare_parameter('joint_names', ['joint1', 'joint2'])
        self.joint_names: List[str] = (
            self.get_parameter('joint_names').get_parameter_value().string_array_value)

        self.declare_parameter('fallback_position', [0.0] * len(self.joint_names))
        self.fallback_position: List[float] = (
            self.get_parameter('fallback_position').get_parameter_value().double_array_value)

        self.get_logger().info("GenIntegrator initialized.")

    def cb_prop(self, msg: JointState) -> None:
        with self._lock:
            conf = self.last_conf

        alpha = self._map_conf_to_alpha(conf)
        u_nom = self._get_nominal()
        u_gen = msg.position

        # 次元チェック
        if len(u_gen) != len(self.joint_names):
            self.get_logger().warn("Joint dimension mismatch.")
            return

        u_safe = self._blend_and_check(u_nom, u_gen, alpha)
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = self.joint_names
        out.position = u_safe if u_safe is not None else self.fallback_position
        self.pub_safe.publish(out)

    def cb_conf(self, msg: Float32) -> None:
        with self._lock:
            self.last_conf = msg.data

    def _map_conf_to_alpha(self, conf: float) -> float:
        # 信頼度を[0,1]にクランプ
        return max(0.0, min(1.0, (conf - 0.2) / 0.8))

    def _get_nominal(self) -> List[float]:
        # 暫定: 静止姿勢を返す
        return list(self.fallback_position)

    def _blend_and_check(self,
                         u_nom: List[float],
                         u_gen: List[float],
                         alpha: float) -> Optional[List[float]]:
        u = [(1.0 - alpha) * n + alpha * g for n, g in zip(u_nom, u_gen)]
        if not self._joint_limits_ok(u):
            return None
        if self._collision_check(u):
            return None
        return u

    def _joint_limits_ok(self, q: List[float]) -> bool:
        # 簡易リミットチェック（±π）
        return all(-3.1416 <= v <= 3.1416 for v in q)

    def _collision_check(self, q: List[float]) -> bool:
        # 実機ではFCL/MoveIt等を使用
        return False


def main(args=None):
    rclpy.init(args=args)
    node = GenIntegrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()