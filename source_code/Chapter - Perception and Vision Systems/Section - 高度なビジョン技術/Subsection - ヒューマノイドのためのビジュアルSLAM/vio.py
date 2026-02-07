import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, JointState
from cv_bridge import CvBridge
import tf_transformations as tf
from typing import Optional, Tuple, List

class VisualInertialOdometry(Node):
    def __init__(self):
        super().__init__('vio_node')
        self.bridge = CvBridge()
        
        # パラメータ
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('joint_topic', '/joint_states')
        
        # ORB特徴量
        self.orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # カメラ内部パラメータ（ROSパラメータサーバから読み込み）
        K_vec = self.declare_parameter('camera_matrix', [0.0]*9).value
        self.K = np.array(K_vec).reshape(3,3)
        
        # 初期化
        self.prev_des: Optional[np.ndarray] = None
        self.prev_kp: Optional[List[cv2.KeyPoint]] = None
        self.prev_depth: Optional[np.ndarray] = None
        self.last_time = self.get_clock().now()
        
        # サブスクライバ
        self.create_subscription(Image, self.get_parameter('camera_topic').value,
                                self.rgb_callback, 10)
        self.create_subscription(Image, self.get_parameter('depth_topic').value,
                                self.depth_callback, 10)
        self.create_subscription(Imu, self.get_parameter('imu_topic').value,
                                self.imu_callback, 100)
        self.create_subscription(JointState, self.get_parameter('joint_topic').value,
                                self.joint_callback, 10)
        
        # バッファ
        self.imu_buffer: List[Imu] = []
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_joints: Optional[JointState] = None
        
    def rgb_callback(self, msg: Image):
        rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        t_cam = rclpy.time.Time.from_msg(msg.header.stamp)
        
        # IMUバッチ取得
        imu_samples = self.get_imu_between(self.last_time, t_cam)
        
        # 関節角度取得
        if self.latest_joints is None:
            return
        theta = np.array(self.latest_joints.position)
        
        # 特徴検出
        kp, des = self.orb.detectAndCompute(rgb, None)
        if self.prev_des is not None and len(kp) > 0:
            matches = self.matcher.match(des, self.prev_des)
            pts2d = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            pts3d = np.float32([self.depth_to_3d(self.prev_kp[m.trainIdx].pt,
                                                 self.prev_depth) for m in matches])
            
            if len(pts3d) >= 6:
                # PnP
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts3d, pts2d, self.K, None, reprojectionError=2.0, confidence=0.99)
                if success and inliers is not None and len(inliers) > 6:
                    T_vis = self.se3_from_rvec_tvec(rvec, tvec)
                    
                    # IMU事前積分
                    delta_T_imu = self.preintegrate_imu(imu_samples)
                    
                    # 運動学ポーズ
                    T_base = self.ekf.state_pose()
                    T_kin = T_base @ self.fk_head(theta) @ self.T_h_c
                    
                    # 融合
                    T_fused = self.fuse_se3([T_vis, T_base @ delta_T_imu, T_kin],
                                           weights=[0.6, 0.3, 0.1])
                    self.ekf.update_pose(T_fused)
        
        # バッファ更新
        self.prev_des, self.prev_kp, self.prev_depth, self.last_time = des, kp, self.latest_depth, t_cam
    
    def depth_callback(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
    
    def imu_callback(self, msg: Imu):
        self.imu_buffer.append(msg)
        # 古いデータ削除
        cutoff = self.get_clock().now() - rclpy.duration.Duration(seconds=1.0)
        self.imu_buffer = [im for im in self.imu_buffer
                          if rclpy.time.Time.from_msg(im.header.stamp) > cutoff]
    
    def joint_callback(self, msg: JointState):
        self.latest_joints = msg
    
    def get_imu_between(self, t_start: rclpy.time.Time, t_end: rclpy.time.Time) -> List[Imu]:
        return [im for im in self.imu_buffer
               if t_start <= rclpy.time.Time.from_msg(im.header.stamp) <= t_end]
    
    def depth_to_3d(self, uv: Tuple[float,float], depth: np.ndarray) -> np.ndarray:
        u, v = int(uv[0]), int(uv[1])
        if 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]:
            z = depth[v,u]
            if z > 0:
                x = (u - self.K[0,2]) * z / self.K[0,0]
                y = (v - self.K[1,2]) * z / self.K[1,1]
                return np.array([x, y, z])
        return np.array([0.0, 0.0, 0.0])
    
    def se3_from_rvec_tvec(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = tvec.flatten()
        return T
    
    def preintegrate_imu(self, imu_samples: List[Imu]) -> np.ndarray:
        # 簡易事前積分（実装は置換可能）
        delta = np.eye(4)
        if len(imu_samples) < 2:
            return delta
        dt = 0.01  # 仮
        for im in imu_samples:
            acc = np.array([im.linear_acceleration.x,
                           im.linear_acceleration.y,
                           im.linear_acceleration.z])
            delta[:3,3] += acc * dt * dt * 0.5
        return delta
    
    def fk_head(self, theta: np.ndarray) -> np.ndarray:
        # 簡易FK（実装は置換可能）
        return np.eye(4)
    
    @property
    def T_h_c(self) -> np.ndarray:
        # 頭部→カメラ変換（ROSパラメータから読み込み）
        return np.array(self.declare_parameter('T_head_camera', np.eye(4).flatten().tolist()).value).reshape(4,4)
    
    def fuse_se3(self, Ts: List[np.ndarray], weights: List[float]) -> np.ndarray:
        # 対数空間で加重平均
        from scipy.spatial.transform import Rotation as R
        ws = np.array(weights)
        ws /= ws.sum()
        t_mean = sum(w * T[:3,3] for w, T in zip(ws, Ts))
        quats = [R.from_matrix(T[:3,:3]).as_quat() for T in Ts]
        # 球面線形補間簡易版
        q_mean = sum(w * q for w, q in zip(ws, quats))
        q_mean /= np.linalg.norm(q_mean)
        T_out = np.eye(4)
        T_out[:3,:3] = R.from_quat(q_mean).as_matrix()
        T_out[:3,3] = t_mean
        return T_out
    
    class EKF:
        def __init__(self):
            self._pose = np.eye(4)
        def state_pose(self) -> np.ndarray:
            return self._pose.copy()
        def update_pose(self, T: np.ndarray):
            self._pose = T.copy()
    
    ekf = EKF()

def main():
    rclpy.init()
    node = VisualInertialOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()