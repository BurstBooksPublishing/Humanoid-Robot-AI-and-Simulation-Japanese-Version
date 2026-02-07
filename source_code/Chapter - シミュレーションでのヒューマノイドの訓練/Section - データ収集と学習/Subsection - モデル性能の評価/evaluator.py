import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_rollouts(csv_path: Path) -> pd.DataFrame:
    """CSVを読み込み、必要な列が揃っているか検証"""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"{csv_path} が見つかりません")
        sys.exit(1)

    required = {"return", "success", "energy", "precision", "recall"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"CSVに必要な列が不足しています: {missing}")
        sys.exit(1)
    return df


def bootstrap_ci(x: np.ndarray, B: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    """Bootstrap で信頼区間を推定"""
    n = len(x)
    rng = np.random.default_rng(42)
    means = np.array([np.mean(rng.choice(x, n, replace=True)) for _ in range(B)])
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """主要指標を算出"""
    metrics = {
        "success_rate": df["success"].mean(),
        "mean_return": df["return"].mean(),
        "energy_per_task": df["energy"].mean(),
        "precision": df["precision"].mean(),
        "recall": df["recall"].mean(),
    }
    # F1 計算（ゼロ除算防止）
    p, r = metrics["precision"], metrics["recall"]
    metrics["f1"] = 2 * p * r / (p + r + 1e-12)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout 評価スクリプト")
    parser.add_argument("csv", type=Path, help="rollouts.csv へのパス")
    parser.add_argument("--output", type=Path, help="JSON 結果保存先（省略時は stdout）")
    args = parser.parse_args()

    df = load_rollouts(args.csv)
    metrics = compute_metrics(df)
    ci_low, ci_high = bootstrap_ci(df["success"].values)

    results = {
        "success_rate": metrics["success_rate"],
        "success_ci": [ci_low, ci_high],
        "mean_return": metrics["mean_return"],
        "energy_per_task": metrics["energy_per_task"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        logger.info(f"結果を {args.output} に保存しました")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()