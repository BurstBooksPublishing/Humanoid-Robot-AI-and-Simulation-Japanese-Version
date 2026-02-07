import math
from typing import Dict, Tuple, Final

# デフォルト値を定数化
DEFAULT_SAFETY: Final[float] = 3.0
DEFAULT_SOLVER_ITERS: Final[int] = 10
DEFAULT_FRICTION: Final[float] = 0.6
DEFAULT_RESTITUTION: Final[float] = 0.05

def compute_timing(
    m_eff: float,
    k_eff: float,
    f_ctrl: float,
    safety_factor: float = DEFAULT_SAFETY
) -> Tuple[float, int, float]:
    """
    臨界タイムステップを計算し、制御周波数に応じたサブステップ数を返す。
    戻り値: (物理ステップ, サブステップ数, 物理周波数)
    """
    if m_eff <= 0.0 or k_eff <= 0.0 or f_ctrl <= 0.0 or safety_factor <= 0.0:
        raise ValueError("全ての引数は正の値である必要があります")

    dt_crit = 2.0 * math.sqrt(m_eff / k_eff)          # 臨界ステップ
    dt_phys = dt_crit / safety_factor                 # 安全マージン適用
    f_phys = 1.0 / dt_phys
    n_sub = max(1, int(math.ceil(f_phys / f_ctrl)))   # 整数サブステップ
    return dt_phys, n_sub, f_phys


def apply_to_simulator(
    sim_config: Dict[str, object],
    dt_phys: float,
    n_sub: int,
    *,
    solver_iterations: int = DEFAULT_SOLVER_ITERS,
    friction: float = DEFAULT_FRICTION,
    restitution: float = DEFAULT_RESTITUTION
) -> Dict[str, object]:
    """
    与えられたシミュレータ設定辞書に物理パラメータを上書きして返す。
    元の辞書は変更されない。
    """
    cfg = sim_config.copy()
    cfg.update({
        "physics_step": dt_phys,
        "substeps": n_sub,
        "solver_iterations": solver_iterations,
        "friction_coefficient": friction,
        "restitution": restitution,
    })
    return cfg


# 使用例
if __name__ == "__main__":
    dt_phys, n_sub, f_phys = compute_timing(
        m_eff=2.0,
        k_eff=1e4,
        f_ctrl=200.0
    )
    sim_cfg = apply_to_simulator({}, dt_phys, n_sub)
    # sim_cfgをシミュレータ初期化へ渡す