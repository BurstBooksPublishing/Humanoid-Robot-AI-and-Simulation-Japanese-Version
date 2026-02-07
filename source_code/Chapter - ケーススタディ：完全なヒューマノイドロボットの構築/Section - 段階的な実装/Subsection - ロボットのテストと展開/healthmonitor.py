#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import time
import threading
from typing import List, Tuple

class HealthMonitor(Node):
    def __init__(self) -> None:
        super().__init__('health_monitor')

        # パラメータ定義
        self.declare_parameter('heartbeat_timeout', 1.0,
                               ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                   description='センサ生存タイムアウト[s]'))
        self.declare_parameter('motor_temp_limit', 75.0,
                               ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                   description='モータ温度上限[℃]'))
        self.declare_parameter('battery_soc_min', 0.2,
                               ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                   description='バッテリ残量下限[0-1]'))

        # QoS：緊急時も到達確実に
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        # パブリッシャ
        self._diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', qos)
        self._status_pub = self.create_publisher(String, '~/status', 10)

        # タイマ
        self._timer = self.create_timer(0.5, self._check_all)

        # 診断データ保持
        self._last_hb_time = time.monotonic()
        self._motor_temp = 0.0
        self._battery_soc = 1.0
        self._sensor_ok = True
        self._motor_ok = True
        self._battery_ok = True
        self._emergency_triggered = False
        self._lock = threading.Lock()

    def _check_all(self) -> None:
        with self._lock:
            now = time.monotonic()
            hb_elapsed = now - self._last_hb_time
            self._motor_temp = self._read_motor_temp()
            self._battery_soc = self._read_battery_soc()

            self._sensor_ok = hb_elapsed < self.get_parameter('heartbeat_timeout').value
            self._motor_ok = self._motor_temp < self.get_parameter('motor_temp_limit').value
            self._battery_ok = self._battery_soc > self.get_parameter('battery_soc_min').value

            status_str = "OK" if all([self._sensor_ok, self._motor_ok, self._battery_ok]) else "FAULT"
            self._status_pub.publish(String(data=status_str))

            # 診断メッセージ生成
            diag = DiagnosticArray()
            diag.header.stamp = self.get_clock().now().to_msg()
            diag.status = [
                DiagnosticStatus(
                    level=DiagnosticStatus.OK if self._sensor_ok else DiagnosticStatus.ERROR,
                    name='health_monitor: sensor_heartbeat',
                    message='Heartbeat alive' if self._sensor_ok else 'Heartbeat lost',
                    values=[KeyValue(key='elapsed_sec', value=str(hb_elapsed))]
                ),
                DiagnosticStatus(
                    level=DiagnosticStatus.OK if self._motor_ok else DiagnosticStatus.WARN,
                    name='health_monitor: motor_temp',
                    message='Temperature normal' if self._motor_ok else 'Overheat',
                    values=[KeyValue(key='temperature_celsius', value=str(self._motor_temp))]
                ),
                DiagnosticStatus(
                    level=DiagnosticStatus.OK if self._battery_ok else DiagnosticStatus.ERROR,
                    name='health_monitor: battery_soc',
                    message='SOC adequate' if self._battery_ok else 'SOC low',
                    values=[KeyValue(key='soc_ratio', value=str(self._battery_soc))]
                )
            ]
            self._diag_pub.publish(diag)

            if status_str == "FAULT" and not self._emergency_triggered:
                self.get_logger().error('致命的な故障、安全停止を発行')
                self._trigger_emergency_stop()
                self._emergency_triggered = True

    # ハードウェアI/O（オーバーライド推奨）
    def _read_sensor_heartbeat(self) -> None:
        # 実機ではセンサから最終受信時刻を更新
        self._last_hb_time = time.monotonic()

    def _read_motor_temp(self) -> float:
        # 実機ではCAN等から取得
        return 45.0

    def _read_battery_soc(self) -> float:
        # 実機ではBMSから取得
        return 0.85

    def _trigger_emergency_stop(self) -> None:
        # 実機ではモータコントローラへ緊急停止コマンド送信
        pass

def main(args=None) -> None:
    rclpy.init(args=args)
    node = HealthMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()