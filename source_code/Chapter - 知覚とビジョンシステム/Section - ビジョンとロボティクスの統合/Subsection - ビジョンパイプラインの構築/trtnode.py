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
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_PATH = '/models/det.trt'
INPUT_SHAPE = (3, 480, 640)   # C, H, W
OBJ_POINTS = np.load('/models/object_points.npy')  # 3D model points
CAM_MATRIX = np.load('/models/camera_matrix.npy')
DIST_COEFFS = np.load('/models/dist_coeffs.npy')


def load_trt_engine(engine_path: str) -> trt.ICudaEngine:
    """TensorRT エンジンを読み込み返す。"""
    if not os.path.isfile(engine_path):
        raise RuntimeError(f'{engine_path} が存在しません')
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


class TrtWrapper:
    """TensorRT 推論ラッパー：メモリ確保〜推論までをカプセル化。"""
    def __init__(self, engine: trt.ICudaEngine):
        self.engine = engine
        self.context = engine.create_execution_context()
        # 入出力バッファ確保
        self.inputs, self.outputs, self.bindings, self.stream = self._alloc_buf()

    def _alloc_buf(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream

    def infer(self, img: np.ndarray) -> np.ndarray:
        """前処理済画像を入力し生の推論結果を返す。"""
        np.copyto(self.inputs[0]['host'], img.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host']


def postprocess(raw: np.ndarray, conf_th: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """生推論結果をフィルタし[(x1,y1,x2,y2), ...]を返す。"""
    # 簡易例：出力が [N,6] (x1,y1,x2,y2,conf,class) と仮定
    raw = raw.reshape(-1, 6)
    keep = raw[:, 4] > conf_th
    return [tuple(map(int, box[:4])) for box in raw[keep]]


def estimate_pose_2d_to_3d(box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """2Dバウンディングボックスから姿勢推定し(rvec, tvec)を返す。"""
    # 簡易：矩形中心を画像座標とし対応する3D点をOBJ_POINTSから選択
    x1, y1, x2, y2 = box
    pts2d = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(
        OBJ_POINTS, pts2d, CAM_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE)
    if not success:
        raise RuntimeError('PnP 失敗')
    return rvec, tvec


def to_pose_msg(rvec: np.ndarray, tvec: np.ndarray, stamp, frame_id: str) -> PoseStamped:
    """rvec/tvecをPoseStampedに変換。"""
    pose = PoseStamped()
    pose.header.stamp = stamp
    pose.header.frame_id = frame_id
    rot_mat, _ = cv2.Rodrigues(rvec)
    qw, qx, qy, qz = cv2.RQDecomp3x3(rot_mat)[1]  # 簡易クォータニオン化
    pose.pose.orientation.w = qw
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = tvec.flatten()
    return pose


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()
        self.trt = TrtWrapper(load_trt_engine(ENGINE_PATH))

        self.sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_cb, 10)
        self.pub = self.create_publisher(PoseStamped, 'vision/pose', 10)

        self.get_logger().info('VisionNode 初期化完了')

    def image_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge 変換失敗: {e}')
            return

        # 前処理
        blob = cv2.dnn.blobFromImage(
            cv_img, scalefactor=1/255.0, size=(INPUT_SHAPE[2], INPUT_SHAPE[1]),
            swapRB=True, crop=False)
        raw = self.trt.infer(blob)
        dets = postprocess(raw)
        if not dets:
            return

        # 先頭検出のみ利用
        rvec, tvec = estimate_pose_2d_to_3d(dets[0])
        pose_msg = to_pose_msg(rvec, tvec, msg.header.stamp, 'base_link')
        self.pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()