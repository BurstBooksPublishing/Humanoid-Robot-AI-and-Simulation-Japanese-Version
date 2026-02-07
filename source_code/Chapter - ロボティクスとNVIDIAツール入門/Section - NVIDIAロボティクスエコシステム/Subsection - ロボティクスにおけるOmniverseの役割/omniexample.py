import random
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import TransformStamped
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.semantics as semantics
from pxr import Usd, UsdGeom, UsdShade, Sdf

class HumanoidSim(Node):
    def __init__(self):
        super().__init__('humanoid_sim')
        self.world = World(stage_units_in_meters=1.0)
        self._setup_scene()
        self._setup_ros()
        self.episode = 0

    def _setup_scene(self):
        # 地面追加
        self.world.scene.add_default_ground_plane()
        # NucleusからUSD取得
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise RuntimeError("Nucleusサーバに接続できません")
        usd_path = assets_root_path + "/Isaac/Robots/Humanoid/humanoid.usd"
        add_reference_to_stage(usd_path, "/World/Humanoid")
        self.humanoid_prim = prim_utils.get_prim_at_path("/World/Humanoid")

        # カメラ生成・胸部に固定
        self.camera = Camera(
            prim_path="/World/CameraDepth",
            resolution=(640, 480),
            translation=(0.1, 0.0, 0.3),  # 胸部からの相対位置
            orientation=np.array([1, 0, 0, 0])
        )
        self.camera.initialize()
        camera_xform = UsdGeom.Xform.Define(self.world.stage, "/World/Humanoid/root/spine/chest/camera_attach")
        self.camera.attach_to_prim(camera_xform.GetPrim())

    def _setup_ros(self):
        self.joint_pub = self.create_publisher(JointState, "/humanoid/joint_states", 10)
        self.depth_pub = self.create_publisher(Image, "/humanoid/depth", 10)
        self.timer = self.create_timer(0.02, self.timer_callback)  # 50Hz

    def randomize_materials(self):
        # マテリアルのラフネスをランダム化
        for prim in self.world.stage.Traverse():
            if prim.IsA(UsdShade.Material):
                roughness_attr = prim.GetAttribute("inputs:roughness")
                if roughness_attr:
                    roughness_attr.Set(random.uniform(0.2, 0.8))

    def timer_callback(self):
        self.world.step(render=True)
        depth = self.camera.get_current_frame()["distance_to_camera"]
        self.publish_depth(depth)
        self.publish_joints()

    def publish_joints(self):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = []
        js.position = []
        for joint_prim in self.humanoid_prim.GetChildren():
            if joint_prim.IsA(UsdGeom.Joint):
                js.name.append(joint_prim.GetName())
                # 簡易：USD attributeから位置取得
                pos_attr = joint_prim.GetAttribute("physics:localPos0")
                if pos_attr:
                    js.position.append(pos_attr.Get())
        self.joint_pub.publish(js)

    def publish_depth(self, depth: np.ndarray):
        img = Image()
        img.header.stamp = self.get_clock().now().to_msg()
        img.height, img.width = depth.shape
        img.encoding = "32FC1"
        img.step = img.width * 4
        img.data = depth.astype(np.float32).tobytes()
        self.depth_pub.publish(img)

    def run_episodes(self, num_episodes=100, steps_per_ep=1000):
        for _ in range(num_episodes):
            self.randomize_materials()
            for _ in range(steps_per_ep):
                rclpy.spin_once(self, timeout_sec=0.0)

def main():
    rclpy.init()
    sim = HumanoidSim()
    sim.run_episodes()
    rclpy.shutdown()

if __name__ == "__main__":
    main()