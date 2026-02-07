import numpy as np
from typing import Tuple, Sequence

# センサモデルの対数オッズ定数
LOG_ODDS_PRIOR: float = 0.0
LOG_ODDS_OCC: float = 0.85
LOG_ODDS_FREE: float = -0.4
LOG_ODDS_MIN: float = -4.0
LOG_ODDS_MAX: float = 4.0


def update_voxel_logodds(
    grid: np.ndarray,
    ray_voxels: Sequence[Tuple[int, int, int]],
    hit: bool,
) -> None:
    """
    3D占有グリッドの対数オッズを、1本のレイで更新する（ベクトル化版）

    Parameters
    ----------
    grid : np.ndarray
        3D配列（shape=(X,Y,Z)）．対数オッズ値を保持．
    ray_voxels : Sequence[Tuple[int,int,int]]
        レイに沿った（終端を除く）ボクセルインデックスのリスト
    hit : bool
        レイが占有セルで終端した場合True
    """
    if not ray_voxels:
        return

    # フリースペース更新（ベクトル化）
    free_delta = LOG_ODDS_FREE - LOG_ODDS_PRIOR
    idx_array = np.asarray(ray_voxels, dtype=int)
    grid[tuple(idx_array.T)] += free_delta
    np.clip(
        grid,
        LOG_ODDS_MIN,
        LOG_ODDS_MAX,
        out=grid,
    )

    # ヒット時のみ終端を占有更新
    if hit:
        end = ray_voxels[-1]
        grid[end] += LOG_ODDS_OCC - LOG_ODDS_PRIOR
        grid[end] = np.clip(grid[end], LOG_ODDS_MIN, LOG_ODDS_MAX)