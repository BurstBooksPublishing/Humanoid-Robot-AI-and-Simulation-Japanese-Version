#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
import re
import threading

MAX_JOINT_VEL = 1.0  # rad/s
MAX_TORQUE = 2.0     # Nm
VEL_REGEX = re.compile(r"vel\s*=\s*([-+]?[0-9]*\.?[0-9]+)")
TORQUE_REGEX = re.compile(r"torque\s*=\s*([-+]?[0-9]*\.?[0-9]+)")

class GuidedLessonBridge(Node):
    def __init__(self):
        super().__init__('guided_lesson_bridge')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub_cmd = self.create_subscription(String, '/student_command', self.cb_cmd, qos)
        self.pub_ctrl = self.create_publisher(String, '/low_level_command', qos)
        self.pub_estop = self.create_publisher(Bool, '/emergency_stop', qos)
        self.sub_joint = self.create_subscription(JointState, '/joint_states', self.cb_joint, qos)
        self.timer = self.create_timer(1.0, self.heartbeat)
        self.last_joint_state = None
        self.lock = threading.Lock()

    def cb_cmd(self, msg: String):
        with self.lock:
            cmd = msg.data.strip()
            if not cmd:
                return
            # 速度リミット適用
            cmd = VEL_REGEX.sub(lambda m: f"vel={min(float(m.group(1)), MAX_JOINT_VEL)}", cmd)
            # トルクリミット適用
            cmd = TORQUE_REGEX.sub(lambda m: f"torque={min(float(m.group(1)), MAX_TORQUE)}", cmd)
            self.pub_ctrl.publish(String(data=cmd))

    def cb_joint(self, msg: JointState):
        with self.lock:
            self.last_joint_state = msg
            # エフォートが閾値超過なら緊急停止
            if msg.effort:
                if any(abs(e) > MAX_TORQUE for e in msg.effort):
                    self.pub_estop.publish(Bool(data=True))

    def heartbeat(self):
        self.get_logger().info('guided lesson active', throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = GuidedLessonBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()