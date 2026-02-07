#!/usr/bin/env python3
# -*- coding: utf-utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.timer import Timer
from rclpy.time import Time
from auction_interfaces.msg import Task, Bid, Winner
from builtin_interfaces.msg import Time as RosTime
import threading
from typing import Dict, List, Optional
import time
import uuid

class AuctionBidder(Node):
    def __init__(self) -> None:
        super().__init__('auction_bidder')
        self.declare_parameter('robot_id', str(uuid.uuid4()))
        self.robot_id: str = self.get_parameter('robot_id').value

        self.declare_parameter('manip_strength', 1.0)
        self.manip_strength: float = self.get_parameter('manip_strength').value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.task_sub = self.create_subscription(
            Task, '/task_auction', self.task_callback, qos)

        self.bid_pub = self.create_publisher(Bid, '/auction_bids', qos)

        self.winner_sub = self.create_subscription(
            Winner, '/auction_winner', self.winner_callback, qos)

        self.pending_tasks: Dict[str, Task] = {}
        self.lock = threading.Lock()

        self.bid_window_sec = 0.5
        self.bid_timer: Optional[Timer] = None

    def task_callback(self, msg: Task) -> None:
        with self.lock:
            self.pending_tasks[msg.task_id] = msg
        # 入札ウィンドウタイマー開始
        if self.bid_timer is None:
            self.bid_timer = self.create_timer(self.bid_window_sec, self.bid_all_pending)

    def bid_all_pending(self) -> None:
        with self.lock:
            tasks = list(self.pending_tasks.values())
            self.pending_tasks.clear()
        for task in tasks:
            bid = self.compute_bid(task)
            self.bid_pub.publish(bid)
        # タイマー停止
        if self.bid_timer:
            self.destroy_timer(self.bid_timer)
            self.bid_timer = None

    def compute_bid(self, task: Task) -> Bid:
        # 推定作業時間を能力で割り、逆数を入札値とする
        est_time = task.workload / (self.manip_strength + 1e-6)
        bid_value = 1.0 / est_time
        bid = Bid()
        bid.task_id = task.task_id
        bid.robot_id = self.robot_id
        bid.bid_value = bid_value
        bid.stamp = self.get_clock().now().to_msg()
        return bid

    def winner_callback(self, msg: Winner) -> None:
        # 自身が落札した場合の処理を追加可能
        pass

def main(args=None) -> None:
    rclpy.init(args=args)
    node = AuctionBidder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()