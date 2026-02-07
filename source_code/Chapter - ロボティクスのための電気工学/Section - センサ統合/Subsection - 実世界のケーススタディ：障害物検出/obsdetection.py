#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
import numpy as np
import open3d as o3d
from sensor_msgs_py import point_cloud2 as pc2
import tf2_ros
from tf2_ros import TransformException
from tf_transformations import quaternion_matrix

# QoS設定：LiDAR/DepthはBestEffortで低遅延
QOS_BEST = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5
)

class ObstacleNode(Node):
    def __init__(self):
        super().__init__('obstacle_node')
        # パラメータ宣言と取得
        self.declare_parameter('voxel_size', 0.03)
        self.declare_parameter('grid_res', 0.05)
        self.declare_parameter('grid_size', 200)
        self.voxel_size = self.get_parameter('voxel_size').value
        self.grid_res = self.get_parameter('grid_res').value
        self.grid_size = self.get_parameter('grid_size').value

        # 購読・配信
        self.sub_lidar = self.create_subscription(
            PointCloud2, '/lidar', self.cb_lidar, QOS_BEST)
        self.sub_depth = self.create_subscription(
            PointCloud2, '/depth', self.cb_depth, QOS_BEST)
        self.pub_grid = self.create_publisher(
            OccupancyGrid, '/obstacle_grid', 10)

        # TFバッファ・リスナ
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 内部地図：log-odds
        self.log_odds = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.l_occ = 0.9
        self.l_free = -0.7

        # タイマーで定期的に地図配信
        self.create_timer(0.1, self.publish_grid)

    def pc2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        # generator→xyz配列へ
        gen = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        return np.array(list(gen))

    def transform_points(self, pts: np.ndarray, frame_id: str, stamp) -> np.ndarray:
        # base_linkへ変換
        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link', frame_id, stamp, timeout=rclpy.duration.Duration(seconds=0.05))
        except TransformException:
            return pts  # 失敗時は変換なし
        q = [trans.transform.rotation.x, trans.transform.rotation.y,
             trans.transform.rotation.z, trans.transform.rotation.w]
        T = quaternion_matrix(q)
        T[0, 3] = trans.transform.translation.x
        T[1, 3] = trans.transform.translation.y
        T[2, 3] = trans.transform.translation.z
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        return (T @ pts_h.T)[:3, :].T

    def cb_lidar(self, msg: PointCloud2):
        pts = self.pc2_to_xyz(msg)
        pts = self.transform_points(pts, msg.header.frame_id, msg.header.stamp)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd = pcd.voxel_down_sample(self.voxel_size)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.02, ransac_n=3, num_iterations=50)
        non_ground = pcd.select_by_index(inliers, invert=True)
        self.update_occupancy(non_ground)

    def cb_depth(self, msg: PointCloud2):
        pts = self.pc2_to_xyz(msg)
        pts = self.transform_points(pts, msg.header.frame_id, msg.header.stamp)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd = pcd.voxel_down_sample(self.voxel_size / 2)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        self.update_occupancy(pcd)

    def update_occupancy(self, pcd: o3d.geometry.PointCloud):
        pts = np.asarray(pcd.points)
        # 2Dグリッド座標へ
        gx = ((pts[:, 0] + self.grid_size * self.grid_res / 2) / self.grid_res).astype(int)
        gy = ((pts[:, 1] + self.grid_size * self.grid_res / 2) / self.grid_res).astype(int)
        mask = (0 <= gx) & (gx < self.grid_size) & (0 <= gy) & (gy < self.grid_size)
        gx, gy = gx[mask], gy[mask]
        # 占有更新
        self.log_odds[gy, gx] += self.l_occ
        # レイキャスティングでfree空間（簡易版）
        for (x, y) in zip(gx, gy):
            self.ray_cast_free(0, 0, x, y)

    def ray_cast_free(self, x0, y0, x1, y1):
        # Bresenhamでfree空間を減算
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if 0 <= x0 < self.grid_size and 0 <= y0 < self.grid_size:
                self.log_odds[y0, x0] += self.l_free
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def publish_grid(self):
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'base_link'
        grid.info = MapMetaData()
        grid.info.resolution = self.grid_res
        grid.info.width = self.grid_size
        grid.info.height = self.grid_size
        grid.info.origin = Pose()
        grid.info.origin.position.x = -self.grid_size * self.grid_res / 2
        grid.info.origin.position.y = -self.grid_size * self.grid_res / 2
        # 確率へ変換
        prob = 1 - 1 / (1 + np.exp(self.log_odds))
        grid.data = (prob * 100).clip(0, 100).astype(np.int8).ravel().tolist()
        self.pub_grid.publish(grid)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()