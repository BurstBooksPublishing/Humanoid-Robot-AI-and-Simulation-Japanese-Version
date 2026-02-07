#!/usr/bin/env python3
import socket
import struct
import time
import logging
import os
from typing import Tuple

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 環境変数から宛先を読み込み、無ければデフォルト
DEST_IP   = os.getenv('TELEM_DEST_IP',   '192.168.1.100')
DEST_PORT = int(os.getenv('TELEM_DEST_PORT', '14550'))
SEND_HZ   = int(os.getenv('TELEM_HZ',        '100'))          # 送信周波数
IMU_DEV   = os.getenv('IMU_DEV',           '/dev/imu')       # IMUデバイスパス

# IMU読み出しダミー（本番は専用ライブラリに置換）
def get_imu() -> Tuple[float, float, float]:
    """IMUから加速度を取得（ダミー実装）"""
    return (0.0, 0.0, 9.81)

def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)                                        # 送信タイムアウト
    addr = (DEST_IP, DEST_PORT)
    seq  = 0
    dt   = 1.0 / SEND_HZ

    logger.info('UDP telemetry started → %s:%d @ %d Hz', *addr, SEND_HZ)

    try:
        while True:
            t0 = time.perf_counter()
            t  = time.time()
            try:
                ax, ay, az = get_imu()
            except Exception as e:
                logger.error('IMU read fail: %s', e)
                continue

            pkt = struct.pack('<Idfff', seq, t, ax, ay, az)
            try:
                sock.sendto(pkt, addr)
            except socket.error as e:
                logger.warning('sendto fail: %s', e)

            seq = (seq + 1) & 0xFFFFFFFF                        # 32bit 循環
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, dt - elapsed))
    except KeyboardInterrupt:
        logger.info('shutting down')
    finally:
        sock.close()

if __name__ == '__main__':
    main()