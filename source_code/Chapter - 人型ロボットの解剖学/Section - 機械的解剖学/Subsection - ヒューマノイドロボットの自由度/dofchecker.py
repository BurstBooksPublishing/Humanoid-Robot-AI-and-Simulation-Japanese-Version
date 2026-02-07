import numpy as np
from typing import Dict, Tuple

# 関節群ごとのDoF定義（実用的なヒューマノイド例）
DOF: Dict[str, int] = {
    'head': 3,
    'torso': 3,
    'arm': 7,
    'leg': 6,
    'hand': 20
}

# 駆動関節総数
N_ACT: int = (
    DOF['head'] +
    DOF['torso'] +
    2 * DOF['arm'] +
    2 * DOF['leg'] +
    DOF['hand']
)

# タスク次元定義（右腕姿勢6, 重心位置3）
TASK_DIMS: Dict[str, int] = {
    'right_arm_pose': 6,
    'com_position': 3
}
M_TASK: int = sum(TASK_DIMS.values())

# 浮遊基底仮想DoF
N_BASE: int = 6

# ヤコビアン形状
J_SHAPE: Tuple[int, int] = (M_TASK, N_ACT)

# 冗長性チェック
REDUNDANCY: int = N_ACT - M_TASK
assert REDUNDANCY >= 0, "タスク次元が駆動DoFを超えている"

# ランク検証用サンプル
J_sample: np.ndarray = np.random.randn(*J_SHAPE)
rank: int = np.linalg.matrix_rank(J_sample)
assert rank == M_TASK, "ヤコビアンがフルランクでない"