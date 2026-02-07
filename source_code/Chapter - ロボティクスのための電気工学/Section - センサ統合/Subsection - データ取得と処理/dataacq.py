import asyncio
import numpy as np
from collections import deque
from typing import Tuple, Optional, Callable
import logging
import time

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# センサデータ型定義
IMUData = np.ndarray  # shape=(6,) [acc, gyro]
JointData = np.ndarray  # shape=(n_joints,)

class SensorBuffer:
    """タイムスタンプ付きセンサデータのリングバッファ"""
    def __init__(self, maxlen: int = 1024):
        self._buf: deque[Tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self._lock = asyncio.Lock()
    
    async def append(self, ts: float, data: np.ndarray):
        async with self._lock:
            self._buf.append((ts, data.copy()))
    
    async def resample(self, t_star: float) -> Optional[np.ndarray]:
        """線形補間によるリサンプリング"""
        async with self._lock:
            buf_list = list(self._buf)
        
        if len(buf_list) < 2:
            return None
            
        for i in range(len(buf_list) - 1):
            t0, x0 = buf_list[i]
            t1, x1 = buf_list[i + 1]
            if t0 <= t_star <= t1:
                alpha = (t_star - t0) / (t1 - t0)
                return x0 + alpha * (x1 - x0)
        return None

class StateEstimator:
    """センサ融合状態推定器"""
    def __init__(self, n_joints: int, control_rate: float):
        self.imu_buf = SensorBuffer()
        self.enc_buf = SensorBuffer()
        self.n_joints = n_joints
        self.dt = 1.0 / control_rate
        self._state_pub: Optional[Callable] = None
    
    def register_publisher(self, pub_func: Callable):
        self._state_pub = pub_func
    
    async def imu_callback(self, ts: float, acc_gyro: IMUData):
        await self.imu_buf.append(ts, acc_gyro)
    
    async def enc_callback(self, ts: float, joint_angles: JointData):
        await self.enc_buf.append(ts, joint_angles)
    
    async def fusion_step(self, t_star: float) -> Optional[np.ndarray]:
        """1ステップのセンサ融合"""
        imu_sample = await self.imu_buf.resample(t_star)
        enc_sample = await self.enc_buf.resample(t_star)
        
        if imu_sample is None or enc_sample is None:
            logger.warning(f"データ欠損 at t={t_star:.3f}")
            return None
        
        # 相補フィルタによる姿勢推定
        fused = self._complementary_filter(imu_sample, enc_sample)
        return fused
    
    def _complementary_filter(self, imu: IMUData, enc: JointData) -> np.ndarray:
        """簡易相補フィルタ: ジャイロ積分 + エンコーダドリフト補正"""
        # 実装例: エンコーダ角度とIMU角速度を融合
        alpha = 0.98  # 補完係数
        gyro_angle = enc + imu[3:6] * self.dt  # ジャイロ積分
        fused_angle = alpha * gyro_angle + (1 - alpha) * enc
        return np.hstack([fused_angle, imu[:3]])  # [角度, 加速度]

async def fusion_loop(estimator: StateEstimator, control_rate: float):
    """非同期センサ融合ループ"""
    dt = 1.0 / control_rate
    t_next = time.time()
    
    while True:
        t_next += dt
        t_star = t_next
        
        try:
            state = await estimator.fusion_step(t_star)
            if state is not None and estimator._state_pub:
                latency = time.time() - t_star
                estimator._state_pub(state, latency)
        except Exception as e:
            logger.error(f"融合エラー: {e}")
        
        sleep_time = max(0, t_next - time.time())
        await asyncio.sleep(sleep_time)

# 使用例
async def main():
    estimator = StateEstimator(n_joints=7, control_rate=100)
    
    # パブリッシャ登録
    def publish_state(state: np.ndarray, latency: float):
        logger.info(f"公開状態: {state[:3]} (遅延: {latency*1000:.1f}ms)")
    
    estimator.register_publisher(publish_state)
    
    # シミュレーションタスク
    async def simulate_sensors():
        t0 = time.time()
        while True:
            t = time.time() - t0
            await estimator.imu_callback(t, np.random.randn(6) * 0.1)
            await estimator.enc_callback(t, np.random.randn(7) * 0.05)
            await asyncio.sleep(0.001)
    
    # 並行実行
    await asyncio.gather(
        fusion_loop(estimator, 100),
        simulate_sensors()
    )

if __name__ == "__main__":
    asyncio.run(main())