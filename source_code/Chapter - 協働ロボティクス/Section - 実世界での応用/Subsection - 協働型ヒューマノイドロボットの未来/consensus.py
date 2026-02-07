import numpy as np
import time
import logging
from typing import List, Tuple

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConsensusController:
    """
    マルチエージェントの合意制御（2D Consensus）
    """
    def __init__(self, adjacency_matrix: np.ndarray, initial_poses: np.ndarray, epsilon: float):
        """
        adjacency_matrix: 隣接行列（N×N）
        initial_poses: 初期位置（N×2）
        epsilon: ステップサイズ（1/d_max未満）
        """
        self.A = adjacency_matrix.astype(float)
        self.poses = initial_poses.astype(float)
        self.eps = epsilon
        self.N = self.A.shape[0]
        
        # ステップサイズの妥当性チェック
        max_degree = np.max(np.sum(self.A, axis=1))
        if self.eps >= 1.0 / max_degree:
            logging.warning(f"ε={self.eps} は理論的上限 {1.0/max_degree:.3f} を超えています")

    def neighbor_states(self, i: int) -> List[np.ndarray]:
        """エージェントiの隣接エージェントの位置を返す"""
        return [self.poses[j] for j in range(self.N) if self.A[i, j] > 0]

    def step(self) -> None:
        """1ステップ分の合意更新"""
        new_poses = self.poses.copy()
        for i in range(self.N):
            nbrs = self.neighbor_states(i)
            if not nbrs:
                continue
            mean_nbr = np.mean(nbrs, axis=0)
            new_poses[i] = self.poses[i] + self.eps * (mean_nbr - self.poses[i])
        self.poses[:] = new_poses

    def run(self, max_steps: int = 1000, sleep_sec: float = 0.01, tol: float = 1e-4) -> Tuple[np.ndarray, int]:
        """
        合意アルゴリズムを実行
        max_steps: 最大ステップ数
        sleep_sec: 通信遅延シミュレーション
        tol: 収束判定閾値
        戻り値: (最終位置, 収束ステップ)
        """
        for k in range(max_steps):
            prev = self.poses.copy()
            self.step()
            time.sleep(sleep_sec)
            # 収束判定：全エージェントの変化量が tol 未満
            if np.max(np.linalg.norm(self.poses - prev, axis=1)) < tol:
                logging.info(f"収束しました (step={k})")
                return self.poses, k
        logging.warning("最大ステップに到達しました")
        return self.poses, max_steps


if __name__ == "__main__":
    # 隣接行列（完全グラフ）
    A = np.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])
    # 初期位置
    poses = np.array([[0.0, 0.0],
                      [1.0, 0.0],
                      [0.5, 0.8]])
    eps = 0.1

    controller = ConsensusController(A, poses, eps)
    final_poses, steps = controller.run(max_steps=1000, sleep_sec=0.01)
    logging.info(f"最終位置:\n{final_poses}")