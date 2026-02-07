#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from your_robotics_lib.contact import ContactMode, ContactSampler
from your_robotics_lib.planning import PRM, RRT, SamplingPlanner
from your_robotics_lib.dynamics import RobotDynamics
from your_robotics_lib.optim import TrajectoryOptimizer, SCPConfig
from your_robotics_lib.utils import rate_limit, nan_guard

# --------------------------------------------------------------------------- #
# ロギング設定
# --------------------------------------------------------------------------- #
_logger = logging.getLogger("task_planner")
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# データ構造
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SubGoal:
    """1サブゴールの記述"""
    name: str
    target_pose: np.ndarray  # 4x4 SE(3)
    contacts: List[ContactMode]


@dataclass(frozen=True)
class LocalPlan:
    """最適化された局所軌道"""
    states: np.ndarray  # (T+1, n)
    controls: np.ndarray  # (T, m)
    contact_sequence: List[ContactMode]
    cost: float


# --------------------------------------------------------------------------- #
# ノード本体
# --------------------------------------------------------------------------- #
class TaskPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("task_planner")

        # ROS 2 パラメータ
        self.declare_parameter(
            "robot_description",
            "",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="URDFファイルパス",
            ),
        )
        self.declare_parameter(
            "max_scp_iters",
            30,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="SCP最大反復数",
            ),
        )
        self.declare_parameter(
            "control_freq",
            100.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="制御周波数[Hz]",
            ),
        )

        # 内部モジュール初期化
        self._dynamics = RobotDynamics(self.get_parameter("robot_description").value)
        self._contact_sampler = ContactSampler(self._dynamics)
        self._sampling_planner: SamplingPlanner = PRM(self._dynamics)
        self._optimizer = TrajectoryOptimizer(
            SCPConfig(max_iters=self.get_parameter("max_scp_iters").value)
        )

        # Publisher / Subscriber
        self._traj_pub = self.create_publisher(
            JointTrajectory, "/local_plan/joint_trajectory", 1
        )
        self.create_subscription(
            PoseArray, "/goal_poses", self._goal_callback, 1
        )

        # 非同期実行用タイマー
        self._timer = self.create_timer(0.1, self._spin_once)

        # 内部状態
        self._goal: Optional[List[SubGoal]] = None
        self._executing: bool = False

    # --------------------------------------------------------------------- #
    # コールバック
    # --------------------------------------------------------------------- #
    def _goal_callback(self, msg: PoseArray) -> None:
        """目標姿勢を受信 → タスク列生成"""
        self._goal = [
            SubGoal(
                name=f"subgoal_{i}",
                target_pose=self._pose_to_se3(pose),
                contacts=self._contact_sampler.sample(pose),
            )
            for i, pose in enumerate(msg.poses)
        ]
        self._executing = True
        _logger.info("新規ゴールを受信、計画開始")

    # --------------------------------------------------------------------- #
    # メインループ
    # --------------------------------------------------------------------- #
    def _spin_once(self) -> None:
        if not self._executing or self._goal is None:
            return

        for subgoal in self._goal:
            local_plan = self._plan_subgoal(subgoal)
            if local_plan is None:
                _logger.warning(f"{subgoal.name} の計画失敗、次を試行")
                continue

            self._publish(local_plan)
            self._wait_execution(local_plan)

        self._executing = False

    # --------------------------------------------------------------------- #
    # 計画関数
    # --------------------------------------------------------------------- #
    def _plan_subgoal(self, subgoal: SubGoal) -> Optional[LocalPlan]:
        """1サブゴールに対して局所最適軌道を生成"""
        # 接触モードを複数サンプリング
        for mode in self._contact_sampler.enumerate(subgoal.contacts):
            # 初期幾何経路生成
            init_path = self._sampling_planner.plan(
                start=self._dynamics.q0,
                goal=subgoal.target_pose,
                mode=mode,
            )
            if init_path is None:
                continue

            # 初期推定値生成
            X0, U0 = self._generate_initial_guess(init_path)

            # SCP 最適化
            Xopt, Uopt = self._optimizer.solve(
                X0, U0,
                dynamics=self._dynamics,
                constraints=self._build_constraints(mode),
                warm_start=True,
            )
            if Xopt is None:
                continue  # 失敗 → 次のモードへ

            nan_guard(Xopt)  # 数値エラー監視
            return LocalPlan(
                states=Xopt,
                controls=Uopt,
                contact_sequence=mode,
                cost=self._optimizer.last_cost,
            )

        return None  # 全モード失敗

    # --------------------------------------------------------------------- #
    # ユーティリティ
    # --------------------------------------------------------------------- #
    @staticmethod
    def _pose_to_se3(pose) -> np.ndarray:
        """geometry_msgs/Pose → SE(3) 4x4"""
        from scipy.spatial.transform import Rotation as R
        T = np.eye(4)
        T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        T[:3, :3] = R.from_quat(
            [pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w]
        ).as_matrix()
        return T

    def _generate_initial_guess(self, path):
        """幾何経路から状態・制御初期推定値を生成"""
        # 簡略化：直線補間 + 逆運動学
        qs = np.array([self._dynamics.ik(p) for p in path])
        xs = np.array([self._dynamics.q2x(q) for q in qs])
        us = np.diff(xs, axis=0) / self._dynamics.dt
        return xs, us

    def _build_constraints(self, mode):
        """接触・衝突回避制約を構築"""
        return {
            "contact": mode,
            "collision": self._dynamics.collision_checker,
        }

    def _publish(self, plan: LocalPlan) -> None:
        """最適軌道をJointTrajectoryとして送信"""
        traj = JointTrajectory()
        traj.header = Header(stamp=self.get_clock().now().to_msg())
        traj.joint_names = self._dynamics.joint_names
        for k, x in enumerate(plan.states):
            pt = JointTrajectoryPoint()
            pt.positions = x[: self._dynamics.nq].tolist()
            pt.velocities = x[self._dynamics.nq:].tolist()
            pt.time_from_start.sec = int(k * self._dynamics.dt)
            traj.points.append(pt)
        self._traj_pub.publish(traj)

    @rate_limit
    def _wait_execution(self, plan: LocalPlan) -> None:
        """実行者が完了するまでスピン（簡略化）"""
        # 実機ではフィードバックトピックを監視
        pass


# --------------------------------------------------------------------------- #
# メイン
# --------------------------------------------------------------------------- #
def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()