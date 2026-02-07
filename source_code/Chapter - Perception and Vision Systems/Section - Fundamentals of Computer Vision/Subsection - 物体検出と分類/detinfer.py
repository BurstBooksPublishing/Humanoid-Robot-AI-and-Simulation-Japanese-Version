import time
import cv2
import numpy as np
import torch
import torchvision
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import threading
import queue

class TrtDetector(Node):
    def __init__(self, engine_path, K, score_th=0.3, nms_iou=0.45, target_size=(640, 384)):
        super().__init__('trt_detector')
        self.engine = self._load_engine(engine_path)  # TensorRTエンジン読み込み
        self.K = np.array(K)
        self.score_th = score_th
        self.nms_iou = nms_iou
        self.target_size = target_size
        self.bridge = CvBridge()
        self.q_rgb = queue.Queue(maxsize=2)
        self.q_depth = queue.Queue(maxsize=2)
        self.pub = self.create_publisher(Detection3DArray, '/detections_3d', 10)
        self.create_subscription(Image, '/camera/color/image_raw', self._cb_rgb, 1)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self._cb_depth, 1)
        threading.Thread(target=self._infer_loop, daemon=True).start()

    def _load_engine(self, path):
        import tensorrt as trt
        with open(path, 'rb') as f, trt.Runtime(trt.Logger()) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _cb_rgb(self, msg):
        if self.q_rgb.full():
            self.q_rgb.get_nowait()
        self.q_rgb.put(self.bridge.imgmsg_to_cv2(msg, 'bgr8'))

    def _cb_depth(self, msg):
        if self.q_depth.full():
            self.q_depth.get_nowait()
        self.q_depth.put(self.bridge.imgmsg_to_cv2(msg, '16UC1'))

    def _preprocess(self, rgb):
        img = cv2.resize(rgb, self.target_size)
        img = img.astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[None, ...]

    def _postprocess(self, boxes, scores, classes):
        keep = scores > self.score_th
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        idxs = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), self.nms_iou)
        return boxes[idxs].numpy(), scores[idxs].numpy(), classes[idxs].numpy()

    def _infer_loop(self):
        while rclpy.ok():
            try:
                rgb = self.q_rgb.get(timeout=1)
                depth = self.q_depth.get(timeout=1)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            x = self._preprocess(rgb)
            boxes, scores, classes = self._infer_trt(x)
            boxes, scores, classes = self._postprocess(boxes, scores, classes)
            msg = Detection3DArray()
            msg.header.stamp = self.get_clock().now().to_msg()
            for b, s, c in zip(boxes, scores, classes):
                u0, v0, u1, v1 = map(int, b)
                z = np.median(depth[v0:v1, u0:u1]) * 0.001  # mm→m
                if np.isfinite(z) and z > 0.2:
                    xyz = z * np.linalg.inv(self.K) @ np.array([(u0 + u1) / 2, (v0 + v1) / 2, 1.0])
                    det = Detection3D()
                    det.bbox.center.position = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(z))
                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = str(int(c))
                    hyp.hypothesis.score = float(s)
                    det.results.append(hyp)
                    msg.detections.append(det)
            self.pub.publish(msg)
            self.get_logger().debug(f'latency: {(time.perf_counter() - t0) * 1000:.1f} ms')

    def _infer_trt(self, x):
        import pycuda.autoinit
        import pycuda.driver as cuda
        with self.engine.create_execution_context() as ctx:
            d_in = cuda.mem_alloc(x.nbytes)
            cuda.memcpy_htod(d_in, x)
            out_shapes = [(100, 4), (100,), (100,)]
            outs = [np.empty(s, dtype=np.float32) for s in out_shapes]
            d_outs = [cuda.mem_alloc(o.nbytes) for o in outs]
            ctx.execute_v2([int(d_in)] + [int(d) for d in d_outs])
            for o, d in zip(outs, d_outs):
                cuda.memcpy_dtoh(o, d)
            return outs

def main():
    rclpy.init()
    K = [[615.0, 0, 320], [0, 615.0, 180], [0, 0, 1]]
    node = TrtDetector('model.trt', K)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()