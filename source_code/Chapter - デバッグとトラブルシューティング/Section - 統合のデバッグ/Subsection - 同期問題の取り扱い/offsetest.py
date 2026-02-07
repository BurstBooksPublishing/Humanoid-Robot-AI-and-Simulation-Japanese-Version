#!/usr/bin/env python3
import numpy as np
import pathlib
import argparse
import logging
from typing import Tuple

# ロギング設定
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_pairs(path: pathlib.Path) -> np.ndarray:
    """CSVから(t_ref, t_loc)を読み込み、shape=(N,2)のndarrayを返す。"""
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] != 2:
        raise ValueError("CSVは2列(t_ref, t_loc)である必要があります")
    return data


def estimate_skew_offset(t_ref: np.ndarray, t_loc: np.ndarray) -> Tuple[float, float, float]:
    """最小二乗法でskew(α)とoffset(β)を推定し、残差の標準偏差も返す。"""
    A = np.vstack([t_ref, np.ones_like(t_ref)]).T
    alpha, beta = np.linalg.lstsq(A, t_loc, rcond=None)[0]
    residuals = t_loc - (alpha * t_ref + beta)
    jitter_std = np.std(residuals)
    return alpha, beta, jitter_std


def windowed_skew(t_ref: np.ndarray, t_loc: np.ndarray, w: int, step: int = 1):
    """スライディングウィンドウで局所skewを推定。"""
    skews = []
    for i in range(0, len(t_ref) - w + 1, step):
        A = np.vstack([t_ref[i : i + w], np.ones(w)]).T
        a, _ = np.linalg.lstsq(A, t_loc[i : i + w], rcond=None)[0]
        skews.append(a)
    return np.array(skews)


def main():
    parser = argparse.ArgumentParser(description="タイムスタンプペアからクロックスキューとオフセットを推定")
    parser.add_argument("csv", type=pathlib.Path, help="timestamp_pairs.csv")
    parser.add_argument("-w", "--window", type=int, default=100, help="ウィンドウ幅（サンプル数）")
    parser.add_argument("-s", "--step", type=int, default=1, help="ウィンドウスライド幅")
    args = parser.parse_args()

    pairs = load_pairs(args.csv)
    t_ref, t_loc = pairs[:, 0], pairs[:, 1]

    alpha, beta, jitter_std = estimate_skew_offset(t_ref, t_loc)
    logging.info(f"alpha(skew) = {alpha:.9f}, beta(offset) = {beta:.6f} s")
    logging.info(f"residual jitter std = {jitter_std * 1e3:.3f} ms")

    skews = windowed_skew(t_ref, t_loc, args.window, args.step)
    drift_ppm = (skews - alpha).mean() * 1e6
    logging.info(f"mean skew drift = {drift_ppm:.2f} ppm")


if __name__ == "__main__":
    main()