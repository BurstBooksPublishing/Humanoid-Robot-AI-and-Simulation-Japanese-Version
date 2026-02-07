import time
import numpy as np
import cv2
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
        self.K = K
        self.score_th = score_th
        self.nms_iou = nms_iou
        self.target_size = target_size
        self.bridge = CvBridge()

        # TensorRT初期化
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger()) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.d_input = cuda.mem_alloc(np.zeros((1,3,*target_size), np.float32).nbytes)
        self.d_output = [cuda.mem_alloc(np.zeros((1, *shape), np.float32).nbytes)
                         for shape in [(25200, 4), (25200,), (25200,)]]  # YOLOv5例
        self.stream = cuda.Stream()

        # 同期キュー
        self.rgb_q = queue.Queue(maxsize=5)
        self.depth_q = queue.Queue(maxsize=5)

        self.pub = self.create_publisher(Detection3DArray, 'detections_3d', 10)
        self.create_subscription(Image, 'rgb', self.rgb_cb, 1)
        self.create_subscription(Image, 'depth', self.depth_cb, 1)
        threading.Thread(target=self.infer_loop, daemon=True).start()

    def rgb_cb(self, msg):
        try:
            self.rgb_q.put(self.bridge.imgmsg_to_cv2(msg, 'bgr8'), block=False)
        except queue.Full:
            pass  # ドロップ

    def depth_cb(self, msg):
        try:
            self.depth_q.put(self.bridge.imgmsg_to_cv2(msg, '32FC1'), block=False)
        except queue.Full:
            pass

    def preprocess(self, rgb):
        img = cv2.resize(rgb, self.target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[None]

    def postprocess(self, boxes, scores, classes):
        keep = scores > self.score_th
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        idx = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), self.nms_iou)
        return boxes[idx].numpy(), scores[idx].numpy(), classes[idx].numpy()

    def infer_loop(self):
        import pycuda.driver as cuda
        while rclpy.ok():
            try:
                rgb = self.rgb_q.get(timeout=0.1)
                depth = self.depth_q.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.time()
            x = self.preprocess(rgb)
            np.copyto(cuda.pagelocked_empty((1,3,*self.target_size), np.float32), x)
            cuda.memcpy_htod_async(self.d_input, x, self.stream)
            self.context.execute_async_v2([int(self.d_input)] + [int(d) for d in self.d_output], self.stream.handle)
            out = []
            for d in self.d_output:
                tmp = np.empty((1, 25200), np.float32)
                cuda.memcpy_dtoh_async(tmp, d, self.stream)
                out.append(tmp)
            self.stream.synchronize()
            boxes, scores, classes = self.postprocess(out[0].reshape(-1,4), out[1].ravel(), out[2].ravel())
            msg = Detection3DArray()
            msg.header.stamp = self.get_clock().now().to_msg()
            for b, s, c in zip(boxes, scores, classes):
                u0, v0, u1, v1 = map(int, b)
                roi = depth[v0:v1, u0:u1]
                z = np.median(roi[np.isfinite(roi)])
                if z > 0.2:
                    xyz = z * np.linalg.inv(self.K) @ np.array([(u0+u1)/2, (v0+v1)/2, 1.0])
                    det = Detection3D()
                    det.results.append(ObjectHypothesisWithPose())
                    det.results[0].hypothesis.class_id = str(int(c))
                    det.results[0].hypothesis.score = float(s)
                    det.bbox.center.position = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
                    msg.detections.append(det)
            self.pub.publish(msg)
            self.get_logger().debug(f'latency: {time.time()-t0:.3f}s')

def main():
    rclpy.init()
    K = np.array([[610, 0, 320], [0, 610, 240], [0, 0, 1]])  # 例
    node = TrtDetector('yolo.engine', K)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()