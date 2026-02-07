import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import threading

class JointCommander(Node):
    def __init__(self):
        super().__init__('joint_commander')

        # パラメータ宣言
        self.declare_parameter(
            'joint_names',
            ['hip', 'knee', 'ankle'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description='制御対象の関節名リスト'
            )
        )
        self.declare_parameter(
            'control_rate',
            50.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='制御周波数[Hz]'
            )
        )
        self.declare_parameter(
            'step_size',
            0.01,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='1ステップあたりの目標位置増分[rad]'
            )
        )

        # QoS設定
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # 購読
        self._sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_cb,
            qos
        )

        # 配信
        self._pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/command',
            qos
        )

        # 状態変数
        self._lock = threading.Lock()
        self._current_positions = []
        self._joint_names = self.get_parameter('joint_names').value

        # タイマー
        period = 1.0 / self.get_parameter('control_rate').value
        self._timer = self.create_timer(period, self._publish_command)

    def _joint_cb(self, msg: JointState):
        with self._lock:
            # 名前順序を揃えて保存
            name_to_pos = dict(zip(msg.name, msg.position))
            self._current_positions = [name_to_pos.get(n, 0.0) for n in self._joint_names]

    def _publish_command(self):
        with self._lock:
            if not self._current_positions:
                return
            step = self.get_parameter('step_size').value
            target = [p + step for p in self._current_positions]

        traj = JointTrajectory()
        traj.joint_names = self._joint_names
        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(self._timer.timer_period_ns)
        traj.points = [point]
        self._pub.publish(traj)

def main(args=None):
    rclpy.init(args=args)
    node = JointCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()