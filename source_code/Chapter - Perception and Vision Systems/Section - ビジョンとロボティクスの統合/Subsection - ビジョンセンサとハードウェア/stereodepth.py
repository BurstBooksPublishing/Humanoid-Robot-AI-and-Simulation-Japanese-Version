import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

# 無効視差マスク値
INVALID_DISPARITY: float = -1.0


def load_rectified_pair(left_path: str, right_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """左右整流済画像をGRAYSCALEで読み込む"""
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise FileNotFoundError("画像読み込み失敗")
    return left, right


def compute_depth_from_disparity(
    disp: np.ndarray, focal_length_px: float, baseline_m: float
) -> np.ndarray:
    """視差→奥行(m)変換。無効視差はNaNにする"""
    depth = np.full_like(disp, np.nan, dtype=np.float32)
    valid = disp > 0.1
    depth[valid] = (focal_length_px * baseline_m) / disp[valid]
    return depth


def main() -> None:
    # パラメータ
    focal_length_px: float = 700.0
    baseline_m: float = 0.12
    out_path: Path = Path("depth_meters.exr")

    # 画像読み込み
    left, right = load_rectified_pair("left_rect.png", "right_rect.png")

    # StereoBMパラメータ（高速・品質バランス）
    stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)
    disp = stereo.compute(left, right).astype(np.float32) / 16.0

    # 視差→奥行
    depth = compute_depth_from_disparity(disp, focal_length_px, baseline_m)

    # 32-bit float EXR保存
    cv2.imwrite(str(out_path), depth)


if __name__ == "__main__":
    main()