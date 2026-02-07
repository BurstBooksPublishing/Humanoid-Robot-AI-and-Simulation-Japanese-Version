import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation

# Isaac Sim起動
sim_app = SimulationApp({"headless": False})

# ROS 2初期化
rclpy.init()
ros_node = Node("isaac_pd_controller")
joint_pub = ros_node.create_publisher(JointState, "joint_states", 10)

world = World(stage_units_in_meters=1.0)
world.reset()
world.scene.add_default_ground_plane()

# ロボット読み込み
humanoid = Articulation(prim_path="/World/Humanoid")
world.scene.add(humanoid)
humanoid.initialize()

# PDゲインと目標姿勢
Kp = 150.0
Kd = 3.5
q_des = humanoid.get_joint_positions()
qd_des = np.zeros_like(q_des)

# JointStateメッセージ準備
joint_names = humanoid.dof_names
joint_state = JointState()
joint_state.name = joint_names

# メインループ
for step in range(10000):
    world.step(render=True)
    q = humanoid.get_joint_positions()
    qd = humanoid.get_joint_velocities()

    # PDトルク計算
    tau = Kp * (q_des - q) + Kd * (qd_des - qd)
    humanoid.set_joint_efforts(tau)

    # ROS 2配信
    joint_state.header.stamp = ros_node.get_clock().now().to_msg()
    joint_state.position = q.tolist()
    joint_state.velocity = qd.tolist()
    joint_pub.publish(joint_state)

    rclpy.spin_once(ros_node, timeout_sec=0.0)

ros_node.destroy_node()
rclpy.shutdown()
sim_app.close()