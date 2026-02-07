#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import JointState
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import threading
import time
from typing import Dict, Optional

# パラメータ記述子（動的再設定対応）
RESID_THRESH_DESC = ParameterDescriptor(
    type=ParameterType.PARAMETER_DOUBLE,
    description='Residual threshold [rad]',
    read_only=False)
TEMP_THRESH_DESC = ParameterDescriptor(
    type=ParameterType.PARAMETER_DOUBLE,
    description='Temperature alert threshold [degC]',
    read_only=False)
ALPHA_DESC = ParameterDescriptor(
    type=ParameterType.PARAMETER_DOUBLE,
    description='EWMA factor',
    read_only=False)

class DiagnosticsNode(Node):
    def __init__(self):
        super().__init__('joint_diagnostics')
        self.declare_parameter('residual_threshold', 0.05, RESID_THRESH_DESC)
        self.declare_parameter('temperature_threshold', 65.0, TEMP_THRESH_DESC)
        self.declare_parameter('ewma_alpha', 0.02, ALPHA_DESC)

        self._resid_thresh = self.get_parameter('residual_threshold').value
        self._temp_thresh = self.get_parameter('temperature_threshold').value
        self._alpha = self.get_parameter('ewma_alpha').value

        self._sub = self.create_subscription(
            JointState, '/joint_states', self._cb_joint, 10)
        self._diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 1)

        self._cmd_pos: Dict[str, float] = {}
        self._ema_temps: Dict[str, float] = {}
        self._lock = threading.Lock()

        # コマンド位置を受け取る（任意）
        self._cmd_sub = self.create_subscription(
            JointState, '/joint_commands', self._cb_cmd, 10)

        # パラメータ変更コールバック
        self.add_on_set_parameters_callback(self._on_params)

    def _on_params(self, params):
        for p in params:
            if p.name == 'residual_threshold':
                self._resid_thresh = p.value
            elif p.name == 'temperature_threshold':
                self._temp_thresh = p.value
            elif p.name == 'ewma_alpha':
                self._alpha = max(0.0, min(1.0, p.value))
        return rclpy.parameter.SetParametersResult(successful=True)

    def _cb_cmd(self, msg: JointState):
        with self._lock:
            for i, name in enumerate(msg.name):
                self._cmd_pos[name] = msg.position[i]

    def _cb_joint(self, msg: JointState):
        now = self.get_clock().now()
        diag_array = DiagnosticArray()
        diag_array.header.stamp = now.to_msg()

        with self._lock:
            for i, name in enumerate(msg.name):
                pos = msg.position[i]
                effort = msg.effort[i] if i < len(msg.effort) else 0.0
                temp = self._estimate_temp(name, effort)

                prev = self._ema_temps.get(name, temp)
                ema = self._alpha * temp + (1.0 - self._alpha) * prev
                self._ema_temps[name] = ema

                cmd = self._cmd_pos.get(name, 0.0)
                residual = abs(pos - cmd)

                status = DiagnosticStatus()
                status.name = f'Joint: {name}'
                status.hardware_id = name

                # レベル判定
                if ema > self._temp_thresh:
                    status.level = DiagnosticStatus.ERROR
                    status.message = 'High motor temperature'
                elif residual > self._resid_thresh:
                    status.level = DiagnosticStatus.WARN
                    status.message = 'Large tracking residual'
                else:
                    status.level = DiagnosticStatus.OK
                    status.message = 'OK'

                status.values = [
                    KeyValue(key='position', value=f'{pos:.4f}'),
                    KeyValue(key='effort', value=f'{effort:.4f}'),
                    KeyValue(key='temperature_estimate', value=f'{ema:.2f}'),
                    KeyValue(key='residual', value=f'{residual:.4f}')
                ]
                diag_array.status.append(status)

        self._diag_pub.publish(diag_array)

    def _estimate_temp(self, name: str, effort: float) -> float:
        # 簡易温度推定：RMS effortに比例
        return 30.0 + 5.0 * abs(effort)

def main(args=None):
    rclpy.init(args=args)
    node = DiagnosticsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()