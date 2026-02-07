import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class StereoDepthEstimator:
    """
    ステレオカメラからの深度推定クラス
    """
    def __init__(self, calib_path: str, max_disp: int = 128):
        # キャリブレーションデータ読み込み
        calib = np.load(calib_path)
        self.left_map1 = calib['left_map1']
        self.left_map2 = calib['left_map2']
        self.right_map1 = calib['right_map1']
        self.right_map2 = calib['right_map2']
        self.Q = calib['Q']
        
        # SGBMマッチャ初期化（人型頭部用パラメータ）
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=max_disp,
            blockSize=5,
            P1=8 * 3 * 5 ** 2,
            P2=32 * 3 * 5 ** 2,
            mode=cv2.STEREO_SGBM_MODE_HH
        )
        
        # WLSフィルタ（平滑化用）
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.matcher)
        self.wls_filter.setLambda(8000)
        self.wls_filter.setSigmaColor(1.5)
        
        # 視差信頼度閾値
        self.conf_thresh = 1.0  # 無効視差閾値
        
    def compute_depth(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ステレオ画像から深度マップを計算
        Returns:
            depth: 奥行き画像（信頼度低いピクセルはnan）
            disp: 視差画像（信頼度低いピクセルはnan）
        """
        # 整直変換
        left_rect = cv2.remap(left_img, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        
        # 視差計算
        disp_left = self.matcher.compute(left_rect, right_rect).astype(np.float32) / 16.0
        
        # 右視差（信頼度マスク用）
        right_matcher = cv2.ximgproc.createRightMatcher(self.matcher)
        disp_right = right_matcher.compute(right_rect, left_rect).astype(np.float32) / 16.0
        
        # WLSフィルタ適用
        disp_filtered = self.wls_filter.filter(disp_left, left_rect, None, disp_right)
        
        # 信頼度マスク生成
        conf_mask = self.wls_filter.getConfidenceMap()
        valid_mask = (disp_filtered > self.conf_thresh) & (conf_mask > 0.0)
        
        # 無効ピクセルをnanに
        disp_filtered[~valid_mask] = np.nan
        
        # 3D再投影
        points = cv2.reprojectImageTo3D(disp_filtered, self.Q)
        depth = points[..., 2]
        
        # 深度無効化
        depth[~valid_mask] = np.nan
        
        return depth, disp_filtered


# 使用例
if __name__ == "__main__":
    estimator = StereoDepthEstimator("stereo_rect_maps.npz")
    
    left = cv2.imread("left.png", cv2.IMREAD_GRAYSCALE)
    right = cv2.imread("right.png", cv2.IMREAD_GRAYSCALE)
    
    depth, disp = estimator.compute_depth(left, right)
    
    # 深度可視化（有効ピクセルのみ）
    valid_depth = depth[~np.isnan(depth)]
    if valid_depth.size > 0:
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("Depth", depth_vis)
        cv2.waitKey(0)