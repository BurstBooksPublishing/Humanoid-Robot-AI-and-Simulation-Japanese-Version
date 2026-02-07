import numpy as np
from typing import List, Tuple, Optional
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LIPMController:
    """Linear Inverted Pendulum Model (LIPM) コントローラ"""
    
    def __init__(self, com_height: float = 0.9, dt: float = 0.02, gravity: float = 9.81):
        self.g = gravity
        self.z_c = com_height
        self.omega2 = self.g / self.z_c
        self.dt = dt
        
        # 状態: [位置, 速度]
        self.x = np.array([0.0, 0.0], dtype=np.float64)
        
        # 歩容パラメータ
        self.footsteps: List[np.ndarray] = []
        self.current_step_index = 0
        
    def set_footsteps(self, footsteps: List[Tuple[float, float]]) -> None:
        """歩容シーケンスを設定"""
        self.footsteps = [np.array(step, dtype=np.float64) for step in footsteps]
        self.current_step_index = 0
        
    def lipm_step(self, zmp_pos: np.ndarray) -> np.ndarray:
        """LIPM ダイナミクスを1ステップ積分"""
        # \ddot{x} = omega^2 * (x - p) の離散化
        accel = self.omega2 * (self.x[0] - zmp_pos[0])
        self.x += self.dt * np.array([self.x[1], accel])
        return self.x.copy()
        
    def get_current_support_foot(self) -> Optional[np.ndarray]:
        """現在の支持脚位置を取得"""
        if self.current_step_index < len(self.footsteps):
            return self.footsteps[self.current_step_index].copy()
        return None
        
    def update_support_foot(self) -> bool:
        """支持脚を次のステップに更新"""
        if self.current_step_index < len(self.footsteps) - 1:
            self.current_step_index += 1
            return True
        return False

def main():
    # コントローラ初期化
    controller = LIPMController(com_height=0.9, dt=0.02)
    
    # 歩容シーケンス設定
    footsteps = [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.6, 0.0)]
    controller.set_footsteps(footsteps)
    
    # シミュレーションパラメータ
    sim_time = 2.0
    total_steps = int(sim_time / controller.dt)
    
    # データ記録用
    com_trajectory = []
    zmp_trajectory = []
    
    # メインループ
    for step in range(total_steps):
        # 現在の支持脚取得
        support_foot = controller.get_current_support_foot()
        if support_foot is None:
            logger.warning("歩容シーケンスが終了しました")
            break
            
        # ZMPを支持脚中心に配置
        zmp_pos = support_foot.copy()
        
        # LIPMダイナミクスを積分
        com_state = controller.lipm_step(zmp_pos)
        
        # データ記録
        com_trajectory.append(com_state.copy())
        zmp_trajectory.append(zmp_pos.copy())
        
        # 10cm進んだら次の支持脚に移行（簡易的な遷移条件）
        if step > 0 and step % 50 == 0:
            controller.update_support_foot()
            
        # ログ出力
        if step % 100 == 0:
            logger.info(f"Step {step}: CoM pos={com_state[0]:.3f}, vel={com_state[1]:.3f}")
    
    # 結果をnumpy配列に変換
    com_trajectory = np.array(com_trajectory)
    zmp_trajectory = np.array(zmp_trajectory)
    
    return com_trajectory, zmp_trajectory

if __name__ == "__main__":
    com_traj, zmp_traj = main()