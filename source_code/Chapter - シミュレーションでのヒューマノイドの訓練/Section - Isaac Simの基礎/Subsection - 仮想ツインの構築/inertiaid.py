import numpy as np
from scipy.optimize import least_squares
from typing import Tuple, Optional, Callable
import logging
import time
import os
import json

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RBDParameterEstimator:
    """剛体ダイナミクスパラメータ推定クラス"""
    
    def __init__(self, 
                 model_func: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                 param_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 cad_param_source: Optional[str] = None):
        """
        Args:
            model_func: RBDモデル関数 (params, q, dq, ddq) -> tau_model
            param_bounds: パラメータの上下限タプル (lower, upper)
            cad_param_source: CADパラメータソースファイルパス
        """
        self.model_func = model_func
        self.param_bounds = param_bounds
        self.cad_param_source = cad_param_source
        
    def residuals(self, params: np.ndarray, q: np.ndarray, dq: np.ndarray, 
                  ddq: np.ndarray, tau_real: np.ndarray) -> np.ndarray:
        """残差計算"""
        try:
            tau_model = self.model_func(params, q, dq, ddq)
            return (tau_real - tau_model).ravel()
        except Exception as e:
            logger.error(f"モデル計算エラー: {e}")
            return np.full(tau_real.size, np.inf)
    
    def validate_data(self, q: np.ndarray, dq: np.ndarray, 
                     ddq: np.ndarray, tau_real: np.ndarray) -> bool:
        """入力データ検証"""
        if not all(isinstance(arr, np.ndarray) for arr in [q, dq, ddq, tau_real]):
            logger.error("入力は全てnumpy.ndarrayである必要があります")
            return False
            
        if not (q.shape == dq.shape == ddq.shape == tau_real.shape):
            logger.error("全ての配列形状が一致している必要があります")
            return False
            
        if q.size == 0:
            logger.error("空の配列が入力されました")
            return False
            
        return True
    
    def estimate(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, 
                 tau_real: np.ndarray, p0: Optional[np.ndarray] = None,
                 **optimizer_kwargs) -> Tuple[np.ndarray, dict]:
        """
        パラメータ推定実行
        
        Returns:
            推定パラメータと最適化結果のメタデータ
        """
        if not self.validate_data(q, dq, ddq, tau_real):
            raise ValueError("無効な入力データです")
        
        # 初期推定値設定
        if p0 is None:
            p0 = self._get_initial_params()
        
        # デフォルト最適化設定
        default_kwargs = {
            'method': 'trf',
            'xtol': 1e-8,
            'ftol': 1e-8,
            'gtol': 1e-8,
            'max_nfev': 2000,
            'verbose': 2,
            'bounds': self.param_bounds
        }
        default_kwargs.update(optimizer_kwargs)
        
        logger.info(f"パラメータ推定開始: 初期パラメータ数={len(p0)}")
        start_time = time.time()
        
        try:
            result = least_squares(
                self.residuals,
                p0,
                args=(q, dq, ddq, tau_real),
                **default_kwargs
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"推定完了: 実行時間={elapsed_time:.2f}s, 反復回数={result.nfev}")
            
            metadata = {
                'success': result.success,
                'cost': result.cost,
                'optimality': result.optimality,
                'nfev': result.nfev,
                'elapsed_time': elapsed_time,
                'message': result.message
            }
            
            return result.x, metadata
            
        except Exception as e:
            logger.error(f"最適化エラー: {e}")
            raise
    
    def _get_initial_params(self) -> np.ndarray:
        """CADまたはデフォルトから初期パラメータ取得"""
        if self.cad_param_source and os.path.exists(self.cad_param_source):
            try:
                with open(self.cad_param_source, 'r') as f:
                    params = json.load(f)
                    return np.array(params['initial_parameters'])
            except Exception as e:
                logger.warning(f"CADパラメータ読み込み失敗: {e}")
        
        # デフォルト: 全パラメータを1.0で初期化
        logger.warning("デフォルトパラメータを使用します")
        return np.ones(10)  # 適切な次元に調整必要
    
    def save_results(self, params: np.ndarray, metadata: dict, 
                    output_path: str) -> None:
        """推定結果をJSON形式で保存"""
        results = {
            'estimated_parameters': params.tolist(),
            'metadata': metadata,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"結果を保存: {output_path}")

# 使用例
def model_torque(params: np.ndarray, q: np.ndarray, dq: np.ndarray, 
                ddq: np.ndarray) -> np.ndarray:
    """
    RBDモデル関数 - 実際の実装に置き換えてください
    この例では簡単な線形モデルを使用
    """
    # ダミー実装 - 実際の剛体ダイナミクスモデルに置き換える
    n_samples, n_joints = q.shape
    tau = np.zeros_like(q)
    
    for i in range(n_joints):
        # 簡略化されたモデル: tau = m*ddq + c*dq + g*q
        m, c, g = params[i*3:(i+1)*3]
        tau[:, i] = m * ddq[:, i] + c * dq[:, i] + g * q[:, i]
    
    return tau

# メイン実行部分
if __name__ == "__main__":
    # データ読み込み（実際のデータソースに置き換える）
    # q, dq, ddq, tau_real = load_measured_data()
    
    # ダミーデータ生成（実際の使用時は削除）
    np.random.seed(42)
    T, n = 1000, 3  # サンプル数、関節数
    q = np.random.randn(T, n)
    dq = np.random.randn(T, n) * 0.1
    ddq = np.random.randn(T, n) * 0.01
    tau_real = model_torque(np.array([1.0, 0.1, 9.8] * n), q, dq, ddq) + np.random.randn(T, n) * 0.05
    
    # 推定器初期化
    estimator = RBDParameterEstimator(
        model_func=model_torque,
        param_bounds=(np.zeros(3*n), np.ones(3*n) * 10)  # パラメータの物理的制約
    )
    
    # パラメータ推定実行
    p_est, metadata = estimator.estimate(q, dq, ddq, tau_real)
    
    # 結果保存
    estimator.save_results(p_est, metadata, "results/estimated_params.json")
    
    # 推定精度評価
    tau_est = model_torque(p_est, q, dq, ddq)
    rmse = np.sqrt(np.mean((tau_real - tau_est)**2))
    logger.info(f"推定RMSE: {rmse:.6f}")