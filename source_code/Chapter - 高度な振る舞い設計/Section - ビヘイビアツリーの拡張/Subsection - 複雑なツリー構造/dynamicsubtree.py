import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32
from ament_index_python import get_package_share_directory

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehaviorTreeNode:
    """ROS 2対応のBehaviorTreeノードラッパー"""
    def __init__(self, node_name: str):
        self.ros_node = Node(node_name)
        self.blackboard = Blackboard()
        self.root = None
        
    def initialize(self):
        """ROS 2ノード初期化"""
        self.mass_sub = self.ros_node.create_subscription(
            Float32, '/detected_payload_mass', self.mass_callback, 10)
        self.strategy_pub = self.ros_node.create_publisher(
            Float32, '/current_strategy', 10)
        
    def mass_callback(self, msg: Float32):
        """ペイロード質量変更時のコールバック"""
        self.adapt_manipulation_strategy(self.root, msg.data)
        
    def spin(self):
        """ROS 2スピンループ"""
        executor = MultiThreadedExecutor()
        executor.add_node(self.ros_node)
        executor.spin()


class Blackboard:
    """スレッドセーフなブラックボード実装"""
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.arbiter = ActuatorArbiter()
        
    def get(self, key: str, default=None):
        return self.data.get(key, default)
        
    def set(self, key: str, value: Any):
        self.data[key] = value


class ActuatorArbiter:
    """アクチュエータアービタ"""
    def __init__(self):
        self._locks: Dict[str, bool] = {}
        
    def lock_for(self, actuators: list):
        """アクチュエータロックコンテキストマネージャー"""
        return ActuatorLock(self, actuators)


class ActuatorLock:
    """アクチュエータロック実装"""
    def __init__(self, arbiter: ActuatorArbiter, actuators: list):
        self.arbiter = arbiter
        self.actuators = actuators
        
    def __enter__(self):
        for actuator in self.actuators:
            if self.arbiter._locks.get(actuator, False):
                raise RuntimeError(f"アクチュエータ {actuator} は既にロックされています")
            self.arbiter._locks[actuator] = True
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        for actuator in self.actuators:
            self.arbiter._locks[actuator] = False


def load_tree(path: str) -> Any:
    """JSONファイルからサブツリーをロード"""
    try:
        full_path = Path(get_package_share_directory('manipulation_strategy')) / path
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return parse_bt_json(data)
    except FileNotFoundError:
        logger.error(f"ファイルが見つかりません: {path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析エラー: {e}")
        raise


def find_parent_and_index(root: Any, target_name: str) -> Tuple[Optional[Any], int]:
    """ターゲットノードの親とインデックスを検索"""
    def _search(node: Any, parent: Any, index: int) -> Tuple[Optional[Any], int]:
        if hasattr(node, 'name') and node.name == target_name:
            return parent, index
        if hasattr(node, 'children'):
            for i, child in enumerate(node.children):
                result = _search(child, node, i)
                if result[0] is not None:
                    return result
        return None, -1
    
    return _search(root, None, -1)


def swap_subtree(root: Any, target_name: str, new_subtree: Any) -> None:
    """ターゲットノードを新しいサブツリーと交換"""
    parent, index = find_parent_and_index(root, target_name)
    if parent is None:
        raise RuntimeError(f"ターゲットノード '{target_name}' が見つかりません")
    
    # アクチュエータロック取得
    with root.blackboard.arbiter.lock_for(new_subtree.required_actuators):
        parent.children[index] = new_subtree
        new_subtree.initialize()
        logger.info(f"サブツリー '{target_name}' を更新しました")


def adapt_manipulation_strategy(root: Any, mass_kg: float) -> None:
    """ペイロード質量に基づいて戦略を適応"""
    if mass_kg > 3.0:
        subtree = load_tree('strategies/heavy_grasp.json')
        logger.info("重量物戦略をロード")
    else:
        subtree = load_tree('strategies/light_grasp.json')
        logger.info("軽量物戦略をロード")
    
    swap_subtree(root, 'manipulation_subtree', subtree)


def sensor_update(blackboard: Blackboard) -> None:
    """センサデータ更新（実装に応じてカスタマイズ）"""
    # 実際のセンサ読み取り処理を実装
    pass


def main():
    """メイン関数"""
    rclpy.init()
    
    bt_node = BehaviorTreeNode('manipulation_strategy_node')
    bt_node.initialize()
    
    # 初期ツリー構築
    bt_node.root = load_tree('trees/main_tree.json')
    
    # 別スレッドでROS 2スピン
    import threading
    ros_thread = threading.Thread(target=bt_node.spin)
    ros_thread.start()
    
    # メインループ
    try:
        rate = bt_node.ros_node.create_rate(100)  # 100 Hz
        while rclpy.ok():
            sensor_update(bt_node.blackboard)
            bt_node.root.tick()
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        bt_node.ros_node.destroy_node()
        rclpy.shutdown()
        ros_thread.join()


if __name__ == '__main__':
    main()