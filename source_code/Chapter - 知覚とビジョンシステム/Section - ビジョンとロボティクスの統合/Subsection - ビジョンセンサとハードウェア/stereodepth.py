import cv2
import numpy as np
from pathlib import Path
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# パラメータ定義
FOCAL_LENGTH_PX = 700.0          # 焦点距離 [px]
BASELINE_M = 0.12                # ベースライン長 [m]
MIN_DISPARITY = 0.1              # 無効視差閾値
STEREO_PARAMS = {
    'numDisparities': 128,
    'blockSize': 15,
    'preFilterType': cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE,
    'preFilterSize': 9,
    'preFilterCap': 31,
    'textureThreshold': 10,
    'uniquenessRatio': 15,
    'speckleWindowSize': 100,
    'speckleRange': 32,
    'disp12MaxDiff': 1
}

def load_rectified_pair(left_path: str, right_path: str) -> tuple[np.ndarray, np.ndarray]:
    """左右整流済画像を読み込む"""
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise FileNotFoundError('画像読み込み失敗')
    return left, right

def compute_depth(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """視差→奥行へ変換"""
    stereo = cv2.StereoBM_create(**STEREO_PARAMS)
    disp = stereo.compute(left, right).astype(np.float32) / 16.0
    valid = disp > MIN_DISPARITY
    depth = np.zeros_like(disp)
    depth[valid] = (FOCAL_LENGTH_PX * BASELINE_M) / disp[valid]
    return depth

def main():
    left_img, right_img = load_rectified_pair('left_rect.png', 'right_rect.png')
    depth_map = compute_depth(left_img, right_img)
    cv2.imwrite('depth_meters.exr', depth_map)
    logging.info('depth_meters.exr 保存完了')

if __name__ == '__main__':
    main()