import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class StereoDepthEstimator:
    """
    ステレオカメラからの深度推定を行うクラス
    """
    
    def __init__(self, calibration_path: str, 
                 num_disparities: int = 128,
                 block_size: int = 5) -> None:
        """
        キャリブレーションデータの読み込みとSGBMマッチャーの初期化
        
        Args:
            calibration_path: ステレオキャリブレーションデータのパス
            num_disparities: 視差数（16の倍数）
            block_size: ブロックサイズ（奇数）
        """
        # キャリブレーションデータの読み込み
        self._load_calibration_data(calibration_path)
        
        # SGBMマッチャーの初期化
        self._init_matcher(num_disparities, block_size)
        
    def _load_calibration_data(self, calibration_path: str) -> None:
        """キャリブレーションデータの読み込み"""
        try:
            maps = np.load(calibration_path)
            self.left_map1 = maps['left_map1']
            self.left_map2 = maps['left_map2']
            self.right_map1 = maps['right_map1']
            self.right_map2 = maps['right_map2']
            self.Q = maps['Q']
        except (FileNotFoundError, KeyError) as e:
            raise ValueError(f"キャリブレーションデータの読み込みに失敗しました: {e}")
    
    def _init_matcher(self, num_disparities: int, block_size: int) -> None:
        """SGBMマッチャーの初期化"""
        if num_disparities % 16 != 0:
            raise ValueError("num_disparitiesは16の倍数である必要があります")
        if block_size % 2 == 0:
            raise ValueError("block_sizeは奇数である必要があります")
            
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            mode=cv2.STEREO_SGBM_MODE_HH
        )
        
    def compute_depth(self, left_img: np.ndarray, right_img: np.ndarray,
                     confidence_threshold: float = 0.6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ステレオ画像から深度マップを計算
        
        Args:
            left_img: 左カメラ画像（グレースケール）
            right_img: 右カメラ画像（グレースケール）
            confidence_threshold: 信頼度閾値
            
        Returns:
            depth: 深度マップ
            disparity: 視差マップ
            confidence_mask: 信頼度マスク
        """
        # 入力検証
        if left_img.shape != right_img.shape:
            raise ValueError("左右画像のサイズが一致しません")
        if len(left_img.shape) != 2 or len(right_img.shape) != 2:
            raise ValueError("入力画像はグレースケールである必要があります")
            
        # 画像の補正
        left_rect = cv2.remap(left_img, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        
        # 視差計算
        disparity = self.matcher.compute(left_rect, right_rect).astype(np.float32) / 16.0
        
        # 3D点群の再投影
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        depth = points_3d[..., 2]
        
        # 信頼度マスクの計算（視差の連続性と有効範囲をチェック）
        confidence_mask = self._compute_confidence_mask(disparity, depth, confidence_threshold)
        
        # 無効な深度値を除外
        depth[~confidence_mask] = np.nan
        
        return depth, disparity, confidence_mask
    
    def _compute_confidence_mask(self, disparity: np.ndarray, depth: np.ndarray,
                                threshold: float) -> np.ndarray:
        """
        視差マップから信頼度マスクを計算
        
        Args:
            disparity: 視差マップ
            depth: 深度マップ
            threshold: 信頼度閾値
            
        Returns:
            信頼度マスク（bool配列）
        """
        # 視差の有効範囲チェック
        valid_disparity = (disparity > 0) & (disparity < self.matcher.getNumDisparities())
        
        # 深度の有効範囲チェック（負の値や極端に大きな値を除外）
        valid_depth = (depth > 0) & (depth < 10.0)  # 10m以内の深度のみ有効
        
        # 視差の連続性チェック（ラプラシアンフィルタでエッジを検出）
        ddx = cv2.Sobel(disparity, cv2.CV_32F, 1, 0, ksize=3)
        ddy = cv2.Sobel(disparity, cv2.CV_32F, 0, 1, ksize=3)
        disparity_smoothness = np.sqrt(ddx**2 + ddy**2)
        smooth_mask = disparity_smoothness < threshold
        
        return valid_disparity & valid_depth & smooth_mask


# 使用例
def main():
    # 深度推定器の初期化
    estimator = StereoDepthEstimator('stereo_rect_maps.npz')
    
    # 画像の読み込み
    left_img = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)
    
    if left_img is None or right_img is None:
        raise FileNotFoundError("画像ファイルが見つかりません")
    
    # 深度マップの計算
    depth, disparity, confidence_mask = estimator.compute_depth(left_img, right_img)
    
    # 結果の保存（オプション）
    cv2.imwrite('depth.png', (depth * 1000).astype(np.uint16))  # mm単位で保存
    cv2.imwrite('disparity.png', (disparity * 16).astype(np.uint16))
    cv2.imwrite('confidence.png', (confidence_mask * 255).astype(np.uint8))


if __name__ == "__main__":
    main()