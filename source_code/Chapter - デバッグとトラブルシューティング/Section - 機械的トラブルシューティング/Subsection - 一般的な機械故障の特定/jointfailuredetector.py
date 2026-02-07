import numpy as np
from typing import Dict, List, Optional

# 閾値と履歴長は外部から上書き可能にする
DEFAULT_WINDOW: int = 100          # 統計計算に使うサンプル数
DEFAULT_TH_POS: float = 0.02       # 位置残差閾値 [rad]
DEFAULT_TH_CURR: float = 0.5       # 電流異常閾値 [A]
DEFAULT_TH_NOISE: float = 0.01     # 標準偏差ベースライン [rad]


class JointFailureDetector:
    """
    関節の位置指令・実測・電流履歴から簡易故障特徴を抽出する
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        th_pos: float = DEFAULT_TH_POS,
        th_curr: float = DEFAULT_TH_CURR,
        th_noise: float = DEFAULT_TH_NOISE,
    ) -> None:
        self.window: int = window
        self.th_pos: float = th_pos
        self.th_curr: float = th_curr
        self.th_noise: float = th_noise

    def detect(
        self,
        cmd_pos_hist: List[float],
        meas_pos_hist: List[float],
        curr_hist: List[float],
    ) -> Dict[str, bool]:
        """
        最新の履歴を受け取り、故障フラグを返す
        """
        # 履長不足なら全フラグFalseで早期リターン
        if len(cmd_pos_hist) < self.window:
            return {"backlash_like": False, "friction_like": False, "intermittent": False}

        # 残差と統計量を計算
        pos_res = np.array(cmd_pos_hist[-self.window :]) - np.array(
            meas_pos_hist[-self.window :]
        )
        pos_mean: float = float(np.mean(pos_res))
        pos_std: float = float(np.std(pos_res))
        curr_mean: float = float(np.mean(curr_hist[-self.window :]))

        # 各故障モードの判定
        flags: Dict[str, bool] = {
            "backlash_like": abs(pos_mean) > self.th_pos and pos_std < self.th_pos,
            "friction_like": curr_mean > self.th_curr and pos_std < self.th_noise,
            "intermittent": pos_std > 3.0 * self.th_noise,
        }
        return flags