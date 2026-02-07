)
        point = JointTrajectoryPoint()
        point.positions = cmd.get('positions', [])
        point.velocities = cmd.get('velocities', [])
        point.effort = cmd.get('effort', [])
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(0.02 * 1e9)
        traj.points.append(point)
        self.joint_pub.publish(traj)

class CheckBalance(py_trees.behaviour.Behaviour):
    def __init__(self, name="CheckBalance", node: IsaacSimBTNode=None):
        super().__init__(name)
        self.node = node
    def update(self):
        if self.node.latest_imu is None:
            return py_trees.common.Status.RUNNING
        q = self.node.latest_imu.orientation
        roll = math.atan2(2.0*(q.w*q.x + q.y*q.z), 1.0 - 2.0*(q.x**2 + q.y**2))
        pitch = math.asin(2.0*(q.w*q.y - q.z*q.x))
        tilt = max(abs(roll), abs(pitch))
        return py_trees.common.Status.SUCCESS if tilt < 0.1 else py_trees.common.Status.FAILURE

class MaintainBalance(py_trees.behaviour.Behaviour):
    def __init__(self, name="MaintainBalance", node: IsaacSimBTNode=None):
        super().__init__(name)
        self.node = node
    def update(self):
        cmd = {
            'joints': ['hip_roll', 'hip_pitch', 'knee'],
            'positions': [0.0, 0.0, 0.0],
            'effort': [5.0, 5.0, 5.0]
        }
        self.node.publish_actuator(cmd)
        return py_trees.common.Status.RUNNING

class RecoveryStep(py_trees.behaviour.Behaviour):
    def __init__(self, name="RecoveryStep", node: IsaacSimBTNode=None):
        super().__init__(name)
        self.node = node
        self.start_time = None
    def initialise(self):
        self.start_time = self.node.get_clock().now()
        cmd = {
            'joints': ['hip_roll', 'hip_pitch', 'knee', 'ankle'],
            'positions': [0.2, -0.3, 0.6, -0.2]
        }
        self.node.publish_actuator(cmd)
    def update(self):
        elapsed = (self.node.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if elapsed > 1.5:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

def main():
    rclpy.init()
    node = IsaacSimBTNode()
    root = py_trees.composites.Selector("Root", memory=False)
    balance_seq = py_trees.composites.Sequence("BalanceSeq", memory=True)
    balance_seq.add_children([
        CheckBalance(node=node),
        MaintainBalance(node=node)
    ])
    root.add_children([balance_seq, RecoveryStep(node=node)])
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup_with_descendants()
    rate = node.create_rate(50)  # 50 Hz
    while rclpy.ok():
        tree.tick()
        rclpy.spin_once(node, timeout_sec=0.0)
        rate.sleep()
    rclpy.shutdown()

if __name__ == '__main__':
    main()