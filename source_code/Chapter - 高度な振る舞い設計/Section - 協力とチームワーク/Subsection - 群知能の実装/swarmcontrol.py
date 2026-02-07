#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from copy import deepcopy
from typing import List, Dict
from geometry_msgs.msg import Twist, Vector3, PoseStamped
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Header
import tf2_ros
import tf2_geometry_msgs
from ament_index_python.packages import get_package_share_directory
import yaml
import os
import time
from threading import Lock

class ConsensusFormationNode(Node):
    def __init__(self):
        super().__init__('consensus_formation_node')
        
        # パラメータ読み込み
        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_id', 0),
                ('alpha', 0.1),
                ('k_r', 1.0),
                ('k_a', 0.5),
                ('control_dt', 0.01),
                ('max_velocity', 0.5),
                ('safety_threshold', 0.1),
                ('formation_radius', 2.0),
                ('neighbor_timeout', 0.5),
            ]
        )
        
        self.robot_id = self.get_parameter('robot_id').value
        self.alpha = self.get_parameter('alpha').value
        self.k_r = self.get_parameter('k_r').value
        self.k_a = self.get_parameter('k_a').value
        self.control_dt = self.get_parameter('control_dt').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_threshold = self.get_parameter('safety_threshold').value
        self.formation_radius = self.get_parameter('formation_radius').value
        self.neighbor_timeout = self.get_parameter('neighbor_timeout').value
        
        # QoS設定
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # パブリッシャー
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos_profile)
        self.state_pub = self.create_publisher(PoseStamped, 'robot_state', qos_profile)
        
        # サブスクライバー
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, qos_profile)
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, qos_profile)
        self.neighbor_sub = self.create_subscription(PoseStamped, 'neighbor_states', self.neighbor_callback, qos_profile)
        
        # TF2バッファ
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 状態変数
        self.state_lock = Lock()
        self.state = {
            'x': np.array([0.0, 0.0]),
            'p': np.array([0.0, 0.0]),
            'p_ref': np.array([0.0, 0.0]),
            'imu': None,
            'joints': None,
            'timestamp': time.time()
        }
        
        self.neighbors: Dict[int, Dict] = {}
        
        # タイマー
        self.control_timer = self.create_timer(self.control_dt, self.control_loop)
        self.state_publish_timer = self.create_timer(0.1, self.publish_state)
        
        self.get_logger().info(f'Robot {self.robot_id} consensus formation node started')

    def imu_callback(self, msg: Imu):
        with self.state_lock:
            self.state['imu'] = msg

    def joint_callback(self, msg: JointState):
        with self.state_lock:
            self.state['joints'] = msg

    def neighbor_callback(self, msg: PoseStamped):
        try:
            # ロボットIDをヘッダーから取得（frame_idに埋め込む想定）
            neighbor_id = int(msg.header.frame_id.split('_')[-1])
            
            # 地面平面座標に変換
            p = np.array([msg.pose.position.x, msg.pose.position.y])
            
            with self.state_lock:
                self.neighbors[neighbor_id] = {
                    'x': np.array([msg.pose.position.x, msg.pose.position.y]),
                    'p': p,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            self.get_logger().warn(f'Failed to process neighbor state: {e}')

    def sense_state(self) -> Dict:
        """現在の状態を取得"""
        with self.state_lock:
            return deepcopy(self.state)

    def get_neighbors(self) -> List[Dict]:
        """有効な近隣ロボットを取得"""
        current_time = time.time()
        valid_neighbors = []
        
        with self.state_lock:
            for neighbor_id, neighbor_data in list(self.neighbors.items()):
                if current_time - neighbor_data['timestamp'] < self.neighbor_timeout:
                    valid_neighbors.append(neighbor_data)
                else:
                    # タイムアウトした近隣を削除
                    del self.neighbors[neighbor_id]
                    
        return valid_neighbors

    def w(self, id1: int, id2: int) -> float:
        """重み関数：距離に基づく重み付け"""
        return 1.0 / (1.0 + abs(id1 - id2))

    def consensus_update(self, state: Dict, neighbors: List[Dict]) -> np.ndarray:
        """コンセンサス更新：式(1)"""
        x = state['x'].copy()
        
        for nb in neighbors:
            # 重み付き平均で共有変数を更新
            x += self.alpha * self.w(self.robot_id, 0) * (nb['x'] - x)
            
        return x

    def formation_force(self, state: Dict, neighbors: List[Dict]) -> np.ndarray:
        """フォーメーション力計算：式(2)"""
        force = np.zeros(2)
        p = state['p']
        
        # 反発力（近づきすぎ防止）
        for nb in neighbors:
            r = p - nb['p']
            d = np.linalg.norm(r) + 1e-6
            
            if d < self.formation_radius:
                # 近距離では強い反発力
                force += self.k_r * r / (d**3)
                
        # 吸引力（目標位置への復帰）
        force += -self.k_a * (p - state['p_ref'])
        
        # 力の大きさを制限
        force_norm = np.linalg.norm(force)
        if force_norm > self.max_velocity * 10:
            force = force / force_norm * self.max_velocity * 10
            
        return force

    def project_to_feasible_motion(self, force: np.ndarray, state: Dict) -> np.ndarray:
        """実行可能な運動に射影（バランス考慮）"""
        # IMUから傾きを取得
        if state['imu'] is not None:
            orientation = state['imu'].orientation
            # 簡易的な傾き計算（実際は適切な変換が必要）
            tilt = np.array([orientation.x, orientation.y])
            
            # 傾きが大きい場合は速度を制限
            tilt_magnitude = np.linalg.norm(tilt)
            if tilt_magnitude > 0.1:
                force *= (1.0 - tilt_magnitude)
                
        # 速度制限
        velocity = force * 0.1  # 力を速度に変換（簡易的）
        velocity_norm = np.linalg.norm(velocity)
        
        if velocity_norm > self.max_velocity:
            velocity = velocity / velocity_norm * self.max_velocity
            
        return velocity

    def safety_check(self, velocity: np.ndarray, state: Dict) -> bool:
        """安全性チェック：ZMP・関節限界考慮"""
        # 簡易的な実装：速度が閾値を超えたら拒否
        if np.linalg.norm(velocity) > self.max_velocity * 1.5:
            return False
            
        # IMUによる転倒検知
        if state['imu'] is not None:
            orientation = state['imu'].orientation
            # 簡易的な転倒判定
            if abs(orientation.x) > 0.3 or abs(orientation.y) > 0.3:
                return False
                
        return True

    def safe_stop_velocity(self) -> np.ndarray:
        """安全停止速度"""
        return np.zeros(2)

    def control_loop(self):
        """メイン制御ループ"""
        try:
            # 状態取得
            state = self.sense_state()
            neighbors = self.get_neighbors()
            
            # コンセンサス更新
            state['x'] = self.consensus_update(state, neighbors)
            
            # フォーメーション力計算
            force = self.formation_force(state, neighbors)
            
            # 実行可能運動に射影
            desired_velocity = self.project_to_feasible_motion(force, state)
            
            # 安全性チェック
            if not self.safety_check(desired_velocity, state):
                desired_velocity = self.safe_stop_velocity()
                self.get_logger().warn('Safety veto activated')
                
            # 速度指令発行
            cmd_vel = Twist()
            cmd_vel.linear.x = desired_velocity[0]
            cmd_vel.linear.y = desired_velocity[1]
            self.cmd_vel_pub.publish(cmd_vel)
            
        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')
            # エラー時は安全停止
            cmd_vel = Twist()
            self.cmd_vel_pub.publish(cmd_vel)

    def publish_state(self):
        """状態を公開（近隣更新用）"""
        try:
            state = self.sense_state()
            
            state_msg = PoseStamped()
            state_msg.header = Header()
            state_msg.header.stamp = self.get_clock().now().to_msg()
            state_msg.header.frame_id = f'robot_{self.robot_id}'
            
            state_msg.pose.position.x = state['x'][0]
            state_msg.pose.position.y = state['x'][1]
            state_msg.pose.position.z = 0.0
            
            self.state_pub.publish(state_msg)
            
        except Exception as e:
            self.get_logger().error(f'State publish error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ConsensusFormationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()