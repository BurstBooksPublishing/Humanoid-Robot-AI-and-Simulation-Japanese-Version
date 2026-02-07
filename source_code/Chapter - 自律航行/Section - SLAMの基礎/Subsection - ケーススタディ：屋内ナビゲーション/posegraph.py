import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, PointCloud2
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import threading
import copy
from typing import Optional

# 独自ライブラリ（GTSAMラッパー、ICP、RGB-Dオドメトリ）を仮定
from gtsam_wrapper import GTSAMWrapper
from rgbd_odometry import run_rgbd_odometry
from scan_matcher import scan_match, loop_candidate, loop_constraint


class PoseGraphUpdater(Node):
    def __init__(self) -> None:
        super().__init__('pose_graph_updater')

        # QoS：ベストエフォート＋適切なキュー深さ
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=200
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # サブスクライバ
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_cb, imu_qos)
        self.rgbd_sub = self.create_subscription(
            PointCloud2, '/camera/depth/points', self.rgbd_cb, sensor_qos)
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/velodyne/points', self.lidar_cb, sensor_qos)

        # パブリッシャ
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/pose_graph/pose', 10)

        # スレッドセーフなバッファとロック
        self.imu_buffer: list[Imu] = []
        self.buffer_lock = threading.Lock()

        # GTSAMラッパー初期化
        self.optimizer = GTSAMWrapper()

        # 最適化実行フラグ
        self.optimize_needed = False
        self.optimize_timer = self.create_timer(0.05, self.try_optimize)  # 20Hz

    # ---- コールバック群 ----
    def imu_cb(self, msg: Imu) -> None:
        with self.buffer_lock:
            self.imu_buffer.append(msg)

    def rgbd_cb(self, msg: PointCloud2) -> None:
        odom = run_rgbd_odometry(msg)
        if odom is not None:
            self.optimizer.add_odometry_factor(odom)
            self.optimize_needed = True

    def lidar_cb(self, msg: PointCloud2) -> None:
        rel = scan_match(msg)
        if rel is None:
            return
        self.optimizer.add_geometry_factor(rel)
        if loop_candidate(rel):
            loop = loop_constraint(rel)
            if loop is not None:
                self.optimizer.add_loop_factor(loop)
        self.optimize_needed = True

    # ---- 最適化 ----
    def try_optimize(self) -> None:
        if not self.optimize_needed:
            return
        self.optimize_needed = False
        self.optimizer.update()
        pose = self.optimizer.get_current_pose()
        self.publish_pose(pose)

    # ---- パブリッシュ ----
    def publish_pose(self, pose) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose = pose  # 仮定：poseはPoseWithCovariance型
        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PoseGraphUpdater()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()