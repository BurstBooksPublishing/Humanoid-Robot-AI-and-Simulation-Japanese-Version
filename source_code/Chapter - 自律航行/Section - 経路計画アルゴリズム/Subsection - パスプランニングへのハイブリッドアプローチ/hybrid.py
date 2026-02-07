#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
import numpy as np
from typing import Optional, List
import time

# 自作モジュール（同一パッケージ内に配置）
from global_planner import AStarPlanner
from local_sampler import KinodynamicRRTStar
from trajectory_optimizer import TrajectoryOptimizer
from mpc_controller import MPCController
from state_estimator import StateEstimator
from obstacle_detector import ObstacleDetector


class HybridPlanner(Node):
    def __init__(self) -> None:
        super().__init__('hybrid_planner')

        # QoS：実機・シミュレータ両対応
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # パラメータ
        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('local_horizon', 5.0)          # [m]
        self.declare_parameter('replan_threshold', 0.3)       # [m] 障害物変化閾値

        self.goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value
        ])

        # プランナ・コントローラ初期化
        self.global_planner = AStarPlanner()
        self.local_sampler = KinodynamicRRTStar()
        self.optimizer = TrajectoryOptimizer()
        self.mpc = MPCController()
        self.state_estimator = StateEstimator()
        self.obstacle_detector = ObstacleDetector()

        # 購読・配信
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/global_path', 10)

        # 内部状態
        self.robot_pose = np.zeros(3)      # [x, y, yaw]
        self.latest_scan: Optional[LaserScan] = None
        self.global_path: Optional[List[np.ndarray]] = None

        # タイマー駆動（100 Hz）
        self.timer = self.create_timer(0.01, self.step)

        self.get_logger().info('Hybrid planner ready.')

    # ---- コールバック ----
    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry) -> None:
        p = msg.pose.pose
        self.robot_pose = np.array([p.position.x, p.position.y, self._yaw_from_quat(p.orientation)])

    # ---- 主ループ ----
    def step(self) -> None:
        if self.latest_scan is None:
            return  # センサ未受信

        # 状態推定更新
        self.state_estimator.update(self.robot_pose, self.latest_scan)

        # ゴール到着判定
        if np.linalg.norm(self.robot_pose[:2] - self.goal) < 0.2:
            self.cmd_pub.publish(Twist())  # 停止
            self.get_logger().info('Goal reached.')
            rclpy.shutdown()
            return

        # グローバル経路生成（初回 or リプラン）
        if self.global_path is None:
            self.global_path = self.global_planner.compute(
                self.robot_pose[:2], self.goal)
            self._publish_path(self.global_path)

        # ローカルウィンドウ抽出
        local_window = self._extract_window(
            self.global_path, self.robot_pose, self.get_parameter('local_horizon').value)

        # 動的障害物変化チェック
        if self.obstacle_detector.significant_change(
                self.latest_scan, self.get_parameter('replan_threshold').value):
            self.global_path = None  # 次ループで再計算
            return

        # ローカル経路生成
        sampled_traj = self.local_sampler.rrt_star(
            local_window, self.robot_pose, self.latest_scan)

        if sampled_traj is None:
            # フォールバック：グローバル経路を単純追従
            control_reference = self._follow_global_coarse(self.global_path)
        else:
            # 最適化 + MPC
            opt_traj = self.optimizer.solve(sampled_traj)
            control_reference = self.mpc.track(opt_traj, self.robot_pose)

        # 指令値送信
        self._send_reference(control_reference)

    # ---- ヘルパ ----
    def _extract_window(self, path: List[np.ndarray], pose: np.ndarray, horizon: float) -> List[np.ndarray]:
        """path 内で現在位置から horizon 以内の点列を返す"""
        dists = [np.linalg.norm(p - pose[:2]) for p in path]
        idx = np.searchsorted(dists, horizon, side='right')
        return path[:idx+1]

    def _follow_global_coarse(self, path: List[np.ndarray]) -> Twist:
        """単純なPure Pursuit風追従（フォールバック用）"""
        if len(path) < 2:
            return Twist()
        target = path[1]
        dx = target[0] - self.robot_pose[0]
        dy = target[1] - self.robot_pose[1]
        theta = self.robot_pose[2]
        e_theta = np.arctan2(dy, dx) - theta
        v = 0.5
        w = 2.0 * np.arctan2(np.sin(e_theta), np.cos(e_theta))
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        return twist

    def _send_reference(self, ref: Twist) -> None:
        self.cmd_pub.publish(ref)

    def _publish_path(self, path: List[np.ndarray]) -> None:
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for p in path:
            ps = PoseStamped()
            ps.pose.position.x = p[0]
            ps.pose.position.y = p[1]
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    @staticmethod
    def _yaw_from_quat(q) -> float:
        import tf_transformations
        return tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]


def main(args=None):
    rclpy.init(args=args)
    node = HybridPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()