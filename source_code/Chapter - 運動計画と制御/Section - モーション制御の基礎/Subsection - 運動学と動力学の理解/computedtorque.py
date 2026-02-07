import numpy as np
import pinocchio as pin
from typing import Optional, Tuple

class ComputedTorqueController:
    def __init__(self, urdf_path: str,
                 kp_diag: np.ndarray,
                 kd_diag: np.ndarray,
                 torque_limits: np.ndarray,
                 tip_ids: Optional[list] = None) -> None:
        # URDF読み込みとデータ生成
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.robot = pin.RobotWrapper(self.model)

        # ゲイン・制限値保存
        self.Kp = np.diag(kp_diag)
        self.Kd = np.diag(kd_diag)
        self.torque_limits = torque_limits

        # 接触先端フレームID（未指定時は両足）
        if tip_ids is None:
            self.tip_ids = [self.model.getFrameId("l_sole"),
                            self.model.getFrameId("r_sole")]
        else:
            self.tip_ids = tip_ids

    def update(self, q: np.ndarray, v: np.ndarray,
               q_des: np.ndarray, v_des: np.ndarray,
               a_des: np.ndarray,
               f_ext: Optional[np.ndarray] = None) -> np.ndarray:
        # 順運動学・動力学一式
        pin.computeAllTerms(self.model, self.data, q, v)
        M = self.data.M
        n = self.model.nv

        # 加速度レベルPD
        e = q_des - q
        ed = v_des - v
        a_ref = a_des + self.Kd @ ed + self.Kp @ e

        # 接触ヤコビアン・外力補償
        J_stack = np.zeros((0, n))
        f_stack = np.zeros(0)
        for fid in self.tip_ids:
            J6 = pin.getFrameJacobian(self.model, self.data, fid,
                                      pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_stack = np.vstack((J_stack, J6))
            if f_ext is not None:
                f_stack = np.hstack((f_stack, f_ext))

        # 計算トルク
        tau = M @ a_ref + self.data.nle
        if f_stack.size:
            tau -= J_stack.T @ f_stack

        # 飽和
        tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        return tau