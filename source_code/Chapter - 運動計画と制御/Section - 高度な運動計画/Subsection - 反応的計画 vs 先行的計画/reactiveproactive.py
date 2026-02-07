import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
import numpy as np
from threading import Lock
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor


class ReactiveMPCNode(Node):
    def __init__(self):
        super().__init__('reactive_mpc_node')

        # QoS設定：センサデータはベストエフォートで最新のみ
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # トピックの購読・配信
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_cb, sensor_qos)
        self.obstacle_sub = self.create_subscription(
            PointCloud2, '/obstacle_fusion', self.obstacle_cb, sensor_qos)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # パラメータ
        self.declare_parameter('T_proactive', 2.0)
        self.declare_parameter('sigma_th', 0.3)
        self.declare_parameter('N', 20)
        self.T_proactive = self.get_parameter('T_proactive').value
        self.sigma_th = self.get_parameter('sigma_th').value
        self.N = self.get_parameter('N').value

        # 共有データ保護
        self.lock = Lock()
        self.x_hat: Optional[np.ndarray] = None
        self.obs: Optional[np.ndarray] = None

        # 非同期MPC用executor
        self.executor = ThreadPoolExecutor(max_workers=1)

        # 周期タイマー（100 Hz）
        self.create_timer(0.01, self.control_loop, callback_group=ReentrantCallbackGroup())

    def odom_cb(self, msg: Odometry):
        with self.lock:
            p = msg.pose.pose
            v = msg.twist.twist
            self.x_hat = np.array([
                p.position.x, p.position.y, p.position.z,
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
                v.linear.x, v.linear.y, v.linear.z
            ])

    def obstacle_cb(self, msg: PointCloud2):
        # PointCloud2 → numpy（簡易実装）
        pts = []
        for i in range(0, len(msg.data), msg.point_step):
            x = np.frombuffer(msg.data[i:i+4], '<f4')[0]
            y = np.frombuffer(msg.data[i+4:i+8], '<f4')[0]
            pts.append([x, y])
        with self.lock:
            self.obs = np.array(pts)

    def get_obstacle_estimates(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.obs

    def get_state_estimate(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.x_hat

    def estimate_ttc(self, x: np.ndarray, obs: np.ndarray) -> float:
        # 簡易TTC：最近傍障害物までの距離 / 速度ノルム
        if obs.size == 0:
            return np.inf
        pos = x[:2]
        vel = x[7:9]
        v_norm = np.linalg.norm(vel)
        if v_norm < 1e-3:
            return np.inf
        dists = np.linalg.norm(obs[:, :2] - pos, axis=1)
        return np.min(dists) / v_norm

    def prediction_uncertainty(self, obs: np.ndarray) -> float:
        # 分散の最大固有値を不確実性指標とする
        if obs.size == 0:
            return 0.0
        cov = np.cov(obs[:, :2].T)
        return np.max(np.linalg.eigvals(cov))

    def reactive_controller(self, x: np.ndarray, obs: np.ndarray) -> np.ndarray:
        # 単純なポテンシャル法による回避
        u = np.zeros(2)
        if obs.size == 0:
            return u
        pos = x[:2]
        for o in obs:
            diff = pos - o[:2]
            d = np.linalg.norm(diff)
            if d < 1e-3:
                continue
            u += diff / d * (1.0 / d**2)
        return np.clip(u, -1.0, 1.0)

    async def solve_mpc_async(self, x: np.ndarray, obs: np.ndarray, horizon: int) -> Optional[np.ndarray]:
        # 非同期MCP（ダミー実装）
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._mpc_solve, x, obs, horizon)

    def _mpc_solve(self, x: np.ndarray, obs: np.ndarray, horizon: int) -> np.ndarray:
        # 実際のMPC求解（ここでは仮に0ベクトル）
        return np.zeros(2)

    def compute_alpha(self, ttc: float, sigma: float) -> float:
        # αはTTCと不確実性に応じて0〜1で変化
        if ttc > self.T_proactive and sigma < self.sigma_th:
            return 0.0
        return 1.0

    def control_loop(self):
        obs = self.get_obstacle_estimates()
        x = self.get_state_estimate()
        if x is None or obs is None:
            return

        t_c = self.estimate_ttc(x, obs)
        sigma = self.prediction_uncertainty(obs)
        u_react = self.reactive_controller(x, obs)

        # MPC非同期実行判定
        if t_c > self.T_proactive and sigma < self.sigma_th:
            u_mpc_future = asyncio.ensure_future(self.solve_mpc_async(x, obs, self.N))
            # 即座に結果を待たず、次ループで利用（簡易）
            u_mpc = None
        else:
            u_mpc = None

        alpha = self.compute_alpha(t_c, sigma)
        u = alpha * u_react + (1 - alpha) * (u_mpc if u_mpc is not None else u_react)

        cmd = Twist()
        cmd.linear.x = u[0]
        cmd.angular.z = u[1]
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = ReactiveMPCNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()