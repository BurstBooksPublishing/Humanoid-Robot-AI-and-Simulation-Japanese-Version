import os
import time
from typing import List, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image

# TensorRT ラッパー（自前実装 or torch2trt 等）
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None
    cuda = None


class TrtWrapper:
    """TensorRT エンジンをロードし、推論を実行する簡易ラッパー"""
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # 入出力バッファ確保
        self.bindings = []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.host_input = host_mem
                self.device_input = device_mem
            else:
                self.host_output = host_mem
                self.device_output = device_mem

    def infer(self, img: np.ndarray) -> np.ndarray:
        # 前処理済み画像をコピー
        np.copyto(self.host_input, img.ravel())
        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()
        return self.host_output


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        self.declare_parameter("model_path", "/models/det.trt")
        self.declare_parameter("input_size", [640, 480])
        self.declare_parameter("conf_thresh", 0.7)

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "camera/image_raw", self.image_cb, 1)
        self.pub = self.create_publisher(PoseStamped, "vision/pose", 10)

        # TensorRT 初期化
        model_path = self.get_parameter("model_path").value
        if not os.path.isfile(model_path):
            self.get_logger().error(f"モデルファイルが見つかりません: {model_path}")
            raise RuntimeError("モデルファイルが見つかりません")
        self.trt = TrtWrapper(model_path)
        self.input_size = tuple(self.get_parameter("input_size").value)
        self.conf_thresh = self.get_parameter("conf_thresh").value

        # カメラ内部パラメータ（仮置き）
        self.K = np.array([[500.0, 0.0, 320.0],
                           [0.0, 500.0, 240.0],
                           [0.0, 0.0, 1.0]])
        self.dist = np.zeros((4, 1))

        # 3D モデル点（単位メートル）
        self.obj_pts = np.array([
            [-0.05, -0.05, 0.0],
            [ 0.05, -0.05, 0.0],
            [ 0.05,  0.05, 0.0],
            [-0.05,  0.05, 0.0]
        ], dtype=np.float32)

        self.get_logger().info("VisionNode 初期化完了")

    def image_cb(self, msg: Image) -> None:
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge 変換エラー: {e}")
            return

        # 前処理
        blob = cv2.resize(cv_img, self.input_size)
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC→CHW
        blob = np.expand_dims(blob, 0)  # 1×C×H×W

        # 推論
        raw = self.trt.infer(blob)  # 例: [1, 6, 8400] → xyxyconf
        dets = self._postprocess(raw)

        if len(dets) == 0:
            return

        # 最も信頼度の高い検出を使用
        best = max(dets, key=lambda x: x[4])
        corners = self._bbox_to_corners(best[:4])

        # PnP
        ok, rvec, tvec = cv2.solvePnP(
            self.obj_pts, corners, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE
        )
        if not ok:
            return

        # PoseStamped 生成
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = "base_link"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = tvec.flatten()
        from geometry_msgs.msg import Quaternion
        q = self._rvec_to_quaternion(rvec)
        pose.pose.orientation = q
        self.pub.publish(pose)

    # 以下、補助メソッド
    def _postprocess(self, raw: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        # 簡易 NMS なし版
        dets = []
        raw = raw.reshape(-1, 6)  # x1,y1,x2,y2,conf,class
        for r in raw:
            if r[4] > self.conf_thresh:
                dets.append(tuple(r[:5]))
        return dets

    def _bbox_to_corners(self, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    def _rvec_to_quaternion(self, rvec: np.ndarray) -> Quaternion:
        from transforms3d.quaternions import quatAboutAxis, normQ
        angle = np.linalg.norm(rvec)
        axis = rvec.flatten() / angle if angle > 1e-9 else [0, 0, 1]
        q = quatAboutAxis(angle, axis)
        q = normQ(q)
        return Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()