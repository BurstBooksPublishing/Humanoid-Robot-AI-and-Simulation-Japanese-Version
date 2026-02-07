import numpy as np
from typing import Optional, Tuple

class HumanoidEKF:
    """
    拡張カルマンフィルタによる人型ロボットの自己位置推定
    状態: [px, py, pz, vx, vy, vz, bax, bay, baz]
    """

    def __init__(self,
                 x0: np.ndarray,
                 P0: np.ndarray,
                 Q: np.ndarray,
                 R_pose: np.ndarray) -> None:
        """
        Args:
            x0: 初期状態ベクトル (9,)
            P0: 初期共分散行列 (9,9)
            Q: プロセスノイズ共分散 (9,9)
            R_pose: 観測ノイズ共分散 (3,3)
        """
        assert x0.shape == (9,), "x0 must be shape (9,)"
        assert P0.shape == (9, 9), "P0 must be shape (9,9)"
        assert Q.shape == (9, 9), "Q must be shape (9,9)"
        assert R_pose.shape == (3, 3), "R_pose must be shape (3,3)"

        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.R_pose = R_pose.copy()

        # 観測行列（位置のみ観測）
        self.H_pose = np.zeros((3, 9))
        self.H_pose[:3, :3] = np.eye(3)

    def predict(self,
                imu_omega: np.ndarray,
                imu_accel: np.ndarray,
                dt: float) -> None:
        """
        IMU情報を用いて状態を予測
        Args:
            imu_omega: 角速度 (3,)
            imu_accel: 加速度 (3,)
            dt: 時刻差 [s]
        """
        assert imu_omega.shape == (3,), "imu_omega must be shape (3,)"
        assert imu_accel.shape == (3,), "imu_accel must be shape (3,)"
        assert dt > 0, "dt must be positive"

        p = self.x[0:3]
        v = self.x[3:6]
        b_a = self.x[6:9]

        # バイアス補正後の加速度
        acc = imu_accel - b_a

        # 状態遷移（等加速度モデル）
        p_new = p + v * dt + 0.5 * acc * dt ** 2
        v_new = v + acc * dt

        self.x[0:3] = p_new
        self.x[3:6] = v_new

        # 線形化された状態遷移行列
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -np.eye(3) * dt

        # 共分散予測
        self.P = F @ self.P @ F.T + self.Q

    def update_pose(self,
                    pose_meas: np.ndarray,
                    R_pose: Optional[np.ndarray] = None) -> None:
        """
        位置観測を用いて状態を更新
        Args:
            pose_meas: 位置観測 [px, py, pz] (3,)
            R_pose: 観測ノイズ共分散 (3,3). Noneの場合はコンストラクタの値を使用
        """
        assert pose_meas.shape == (3,), "pose_meas must be shape (3,)"

        R = R_pose if R_pose is not None else self.R_pose

        # 予測観測
        z_pred = self.H_pose @ self.x

        # イノベーション共分散
        S = self.H_pose @ self.P @ self.H_pose.T + R

        # カルマンゲイン
        K = self.P @ self.H_pose.T @ np.linalg.inv(S)

        # 状態更新
        self.x += K @ (pose_meas - z_pred)

        # 共分散更新（Joseph form for numerical stability）
        I_KH = np.eye(9) - K @ self.H_pose
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        現在の状態と共分散を取得
        Returns:
            x: 状態ベクトル (9,)
            P: 共分散行列 (9,9)
        """
        return self.x.copy(), self.P.copy()