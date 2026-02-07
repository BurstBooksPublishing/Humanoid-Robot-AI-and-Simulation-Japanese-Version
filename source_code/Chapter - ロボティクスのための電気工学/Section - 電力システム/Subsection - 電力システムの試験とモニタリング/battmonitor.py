#!/usr/bin/env python3
import time
import logging
from dataclasses import dataclass
from typing import Callable

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bms")

# 定数
NOMINAL_CAPACITY_AH = 50.0          # 公称容量
DT_S = 0.1                          # 積分周期
VOLTAGE_MIN_V = 42.0                # 下限電圧
TEMP_MAX_C = 65.0                   # 上限温度
SOC_MIN = 0.05                      # 下限SoC
CURRENT_THRESHOLD_A = 0.5           # 低電流判定閾値
OCV_SLOPE = 0.4                     # OCV線形近似傾き
OCV_INTERCEPT_V = 3.6               # OCV線形近似切片
SOC_FILTER_GAIN = 0.1               # 電圧補正ゲイン

# ハードウェアI/F（本番ではドライバに置換）
def read_current() -> float: return 12.3  # A
def read_voltage() -> float: return 48.0  # V
def read_temp() -> float: return 35.0     # °C

@dataclass
class BmsState:
    soc: float

class BmsCore:
    def __init__(self, capacity_ah: float, initial_soc: float):
        self.capacity_c = capacity_ah * 3600.0  # クーロン変換
        self.state = BmsState(soc=max(0.0, min(1.0, initial_soc)))

    def update(self, i_a: float, v_v: float, dt_s: float) -> None:
        # クーロン計数
        self.state.soc -= (i_a * dt_s) / self.capacity_c
        self.state.soc = max(0.0, min(1.0, self.state.soc))

        # 低電流時のOCV補正
        if abs(i_a) < CURRENT_THRESHOLD_A:
            voc_per_cell = OCV_INTERCEPT_V + OCV_SLOPE * self.state.soc
            measured_soc = (v_v / 16.0 - OCV_INTERCEPT_V) / OCV_SLOPE  # 16Sパック想定
            measured_soc = max(0.0, min(1.0, measured_soc))
            self.state.soc = (1.0 - SOC_FILTER_GAIN) * self.state.soc + SOC_FILTER_GAIN * measured_soc

    def check_alarm(self, v_v: float, t_c: float) -> bool:
        return v_v < VOLTAGE_MIN_V or t_c > TEMP_MAX_C or self.state.soc < SOC_MIN

def main() -> None:
    bms = BmsCore(NOMINAL_CAPACITY_AH, 0.9)
    while True:
        i = read_current()
        v = read_voltage()
        t = read_temp()
        bms.update(i, v, DT_S)
        if bms.check_alarm(v, t):
            logger.warning("ALARM V=%.2f I=%.2f T=%.1f SoC=%.3f", v, i, t, bms.state.soc)
        time.sleep(DT_S)

if __name__ == "__main__":
    main()