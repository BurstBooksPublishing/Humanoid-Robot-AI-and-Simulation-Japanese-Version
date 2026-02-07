import math
from typing import Tuple, List

# パラメータをクラスで管理（保守性向上）
class FingerParams:
    l1: float = 0.05          # リンク1長さ[m]
    l2: float = 0.03          # リンク2長さ[m]
    gear_ratio: float = 50.0  # 減速比
    efficiency: float = 0.85  # 伝達効率

def compute_jacobian(theta1: float, theta2: float, params: FingerParams) -> List[List[float]]:
    """ヤコビ行列を計算"""
    s1 = math.sin(theta1)
    s12 = math.sin(theta1 + theta2)
    c1 = math.cos(theta1)
    c12 = math.cos(theta1 + theta2)

    J = [
        [-params.l1 * s1 - params.l2 * s12, -params.l2 * s12],
        [ params.l1 * c1 + params.l2 * c12,  params.l2 * c12]
    ]
    return J

def compute_motor_torques(
    theta1: float,
    theta2: float,
    F_tip: float,
    params: FingerParams
) -> Tuple[float, float, float, float]:
    """
    関節トルクとモータ必要トルクを返す
    F_tip: 指先に垂直に作用する力[N]
    """
    J = compute_jacobian(theta1, theta2, params)
    Fx, Fy = 0.0, F_tip

    # tau = J^T * F
    tau1 = J[0][0] * Fx + J[1][0] * Fy
    tau2 = J[0][1] * Fx + J[1][1] * Fy

    # 減速機・効率を考慮
    motor_tau1 = abs(tau1) / (params.gear_ratio * params.efficiency)
    motor_tau2 = abs(tau2) / (params.gear_ratio * params.efficiency)

    return tau1, tau2, motor_tau1, motor_tau2

def main() -> None:
    params = FingerParams()
    theta1 = math.radians(30)
    theta2 = math.radians(20)
    F_tip = 10.0

    tau1, tau2, mt1, mt2 = compute_motor_torques(theta1, theta2, F_tip, params)

    print(f"{tau1=:.3f}, {tau2=:.3f}")   # 関節トルク[Nm]
    print(f"{mt1=:.3f}, {mt2=:.3f}")     # モータ必要トルク[Nm]

if __name__ == "__main__":
    main()