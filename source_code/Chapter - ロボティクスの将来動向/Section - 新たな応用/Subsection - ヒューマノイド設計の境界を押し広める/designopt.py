import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, Any
import logging

# ログ設定（本番環境では外部設定ファイルから読み込む）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatterySizingOptimizer:
    """バッテリ質量最適化クラス：ミッション要求を満たしながら総質量を最小化"""
    
    def __init__(self,
                 avg_power_w: float = 150.0,
                 mission_time_s: float = 7200.0,
                 energy_density_wh_per_kg: float = 180.0,
                 base_struct_mass_kg: float = 20.0,
                 struct_mass_factor: float = 0.3,
                 payload_mass_kg: float = 5.0,
                 batt_mass_bounds_kg: Tuple[float, float] = (1.0, 200.0)) -> None:
        """
        パラメータはデフォルト値を持たせつつ外部注入可能にして単体テストしやすくする
        """
        self.P_avg = avg_power_w
        self.T_mission = mission_time_s
        self.e_batt_J = energy_density_wh_per_kg * 3600.0
        self.m_struct_base = base_struct_mass_kg
        self.m_struct_factor = struct_mass_factor
        self.m_payload = payload_mass_kg
        self.batt_bounds = batt_mass_bounds_kg
        
        # 要求エネルギー事前計算
        self.E_req = self.P_avg * self.T_mission
        
        # 最適化結果キャッシュ
        self._result: Dict[str, Any] = {}
    
    def total_mass(self, m_batt: float) -> float:
        """構造質量をバッテリ質量に比例させて推定"""
        m_struct = self.m_struct_base + self.m_struct_factor * m_batt
        return m_batt + m_struct + self.m_payload
    
    def energy_constraint(self, x: np.ndarray) -> float:
        """エネルギー制約：使用可能エネルギー - 要求エネルギー >= 0"""
        m_batt = x[0]
        E_avail = m_batt * self.e_batt_J
        return E_avail - self.E_req
    
    def objective(self, x: np.ndarray) -> float:
        """目的関数：総質量最小化"""
        return self.total_mass(x[0])
    
    def optimize(self) -> Dict[str, Any]:
        """最適化実行：失敗時は例外を送出して呼び出し側でハンドリング"""
        x0 = np.array([(self.batt_bounds[0] + self.batt_bounds[1]) / 2])
        bounds = [self.batt_bounds]
        cons = ({'type': 'ineq', 'fun': self.energy_constraint})
        
        try:
            res = minimize(
                self.objective,
                x0,
                bounds=bounds,
                constraints=cons,
                method='SLSQP',
                options={'ftol': 1e-6, 'disp': False}
            )
            if not res.success:
                raise RuntimeError(f"Optimization failed: {res.message}")
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            raise
        
        m_batt_opt = float(res.x[0])
        m_total_opt = self.total_mass(m_batt_opt)
        
        self._result = {
            'battery_mass_kg': m_batt_opt,
            'total_mass_kg': m_total_opt,
            'struct_mass_kg': self.m_struct_base + self.m_struct_factor * m_batt_opt,
            'energy_margin_J': self.energy_constraint(res.x),
            'success': True
        }
        logger.info(f"Optimization complete: {self._result}")
        return self._result

def main() -> None:
    """CLIエントリポイント"""
    optimizer = BatterySizingOptimizer()
    result = optimizer.optimize()
    print(f"Optimal battery mass: {result['battery_mass_kg']:.2f} kg")
    print(f"Total system mass: {result['total_mass_kg']:.2f} kg")

if __name__ == "__main__":
    main()