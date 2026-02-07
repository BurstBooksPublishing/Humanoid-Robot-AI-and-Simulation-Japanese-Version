import time
import logging
from typing import Dict, Any, Optional

# ROS 2 用のインポート（必要に応じてコメントアウトを外す）
# import rclpy
# from rclpy.node import Node

# タイムア�ト閾値（秒）
TIMEOUT: float = 0.05

# 統計情報を保持する辞書
stats: Dict[str, Dict[str, Any]] = {}


def profile_tick(node) -> str:
    """
    ノードの実行時間を計測し、統計情報を更新する。
    タイムアウトを超えた場合は警告を出してノードを中断する。
    """
    start: float = time.perf_counter()
    status: str = node.tick()  # ノードの実行
    elapsed: float = time.perf_counter() - start

    # 統計情報の更新
    s: Dict[str, Any] = stats.setdefault(node.name, {
        'count': 0,
        'total': 0.0,
        'max': 0.0,
        'succ': 0
    })
    s['count'] += 1
    s['total'] += elapsed
    s['max'] = max(s['max'], elapsed)
    if status == 'SUCCESS':
        s['succ'] += 1

    # タイムアウト検出
    if elapsed > TIMEOUT:
        logging.warning(
            f"Node {node.name} exceeded timeout: {elapsed:.3f}s > {TIMEOUT}s"
        )
        # ノードに中断メソッドがあれば呼び出す
        if hasattr(node, 'abort') and callable(getattr(node, 'abort')):
            node.abort()

    return status


def reset_stats() -> None:
    """統計情報をリセットする"""
    global stats
    stats.clear()


def get_stats(node_name: Optional[str] = None) -> Dict[str, Any]:
    """
    指定されたノードの統計情報を返す。
    node_name が None の場合は全ノードの統計を返す。
    """
    if node_name is None:
        return stats
    return stats.get(node_name, {})


# 使用例（ROS 2 ノード内での利用を想定）
# class ProfiledBehaviorTree(Node):
#     def __init__(self):
#         super().__init__('profiled_behavior_tree')
#         self.tree = ...  # ビヘイビアツリーの構築
#
#     def tick(self):
#         for node in self.tree.nodes:
#             profile_tick(node)