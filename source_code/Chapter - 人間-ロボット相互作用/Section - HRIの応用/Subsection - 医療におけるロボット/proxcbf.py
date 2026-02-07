import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import numpy as np
from typing import Optional
import osqp
from scipy import sparse


class ProxemicCBF(Node):
    def __init__(self) -> None:
        super().__init__('proxemic_cbf')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub_pose = self.create_subscription(
            Float32, 'human_distance', self.pose_cb, qos)
        self.sub_affect = self.create_subscription(
            Float32, 'human_discomfort', self.affect_cb, qos)

        self.pub_cmd = self.create_publisher(Twist, 'cmd_vel', qos)

        self.timer = self.create_timer(0.01, self.control_loop)

        self.d: float = 2.0
        self.e: float = 0.0
        self.v_max: float = 1.0
        self.v_min: float = 0.0
        self.d_safe: float = 0.6
        self.kappa: float = 3.0
        self.lambda_c: float = 0.5

    def pose_cb(self, msg: Float32) -> None:
        self.d = msg.data

    def affect_cb(self, msg: Float32) -> None:
        self.e = msg.data

    def control_loop(self) -> None:
        v_des = 0.5
        C = 1.0 / (1.0 + self.d) + self.e
        h = self.d - self.d_safe

        # OSQPで解くQP: 1/2 u^T P u + q^T u  s.t. l <= A u <= u
        P = sparse.csc_matrix([[1.0 + self.lambda_c * C]])
        q = np.array([-v_des])
        A_sparse = sparse.csc_matrix([[-1.0]])
        l = np.array([-np.inf])
        u = np.array([-self.kappa * h])

        prob = osqp.OSQP()
        prob.setup(P, q, A_sparse, l, u, verbose=False)
        res = prob.solve()

        if res.info.status != 'solved':
            self.get_logger().warn('QP failed')
            return

        u_opt = np.clip(res.x[0], self.v_min, self.v_max)

        cmd = Twist()
        cmd.linear.x = float(u_opt)
        self.pub_cmd.publish(cmd)


def main() -> None:
    rclpy.init()
    node = ProxemicCBF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()