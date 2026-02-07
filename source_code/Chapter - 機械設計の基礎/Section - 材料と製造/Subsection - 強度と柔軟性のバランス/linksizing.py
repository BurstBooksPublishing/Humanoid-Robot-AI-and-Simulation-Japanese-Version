import numpy as np
from typing import Final, Tuple

# 材料・幾何パラメータ
M: Final[float] = 50.0          # N·m
L: Final[float] = 0.30          # m
b: Final[float] = 0.02          # m
E: Final[float] = 70e9          # Pa
rho: Final[float] = 2700        # kg/m^3
sigma_allow: Final[float] = 250e6  # Pa
K: Final[float] = 1.0           # オイラー係数
P: Final[float] = 200.0         # N

def required_thickness(moment: float, width: float, sigma_max: float) -> float:
    """曲げ応力が許容値以下となる最小板厚を返す"""
    return np.sqrt(6.0 * moment / (width * sigma_max))

def euler_critical_load(width: float, thickness: float, length: float,
                        young: float, k_factor: float) -> float:
    """矩形断面のオイラー座屈荷重"""
    I = width * thickness**3 / 12.0
    return np.pi**2 * young * I / (k_factor * length)**2

def mass_estimate(width: float, thickness: float, length: float, density: float) -> float:
    """見かけ質量（自重は無視）"""
    return width * thickness * length * density

def main() -> None:
    t_req = required_thickness(M, b, sigma_allow)
    Pcr = euler_critical_load(b, t_req, L, E, K)
    m = mass_estimate(b, t_req, L, rho)

    print(f"必要厚さ t = {t_req*1000:.2f} mm")
    print(f"オイラー座屈 Pcr = {Pcr:.1f} N, 安全率 = {Pcr/P:.1f}")
    print(f"推定質量 m = {m*1000:.2f} g")

if __name__ == "__main__":
    main()