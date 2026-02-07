#!/usr/bin/env python3
import socket
import struct
import time
import zlib
import logging
from typing import Tuple

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StateBroadcaster:
    def __init__(self, sender_id: int, addr: Tuple[str, int], hz: int = 100):
        self.sender_id = sender_id
        self.addr = addr
        self.hz = hz
        self.seq = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.settimeout(0.01)  # 送信タイムアウト
        logger.info(f"Broadcaster initialized: id={sender_id}, addr={addr}, hz={hz}")

    def pack_state(self, pose: Tuple[float, float, float], joint_vels: Tuple[float, ...]) -> bytes:
        ts = int(time.time_ns() // 1000)  # マイクロ秒タイムスタンプ
        payload = struct.pack('fff', *pose)
        payload += struct.pack(f'{len(joint_vels)}f', *joint_vels[:3])  # 最大3軸送信
        header = struct.pack('!I Q H B H', self.seq, ts, self.sender_id, 1, len(payload))
        checksum = zlib.crc32(header + payload) & 0xffffffff
        self.seq = (self.seq + 1) & 0xffffffff
        return header + payload + struct.pack('!I', checksum)

    def send(self, pose: Tuple[float, float, float], joint_vels: Tuple[float, ...]) -> bool:
        try:
            msg = self.pack_state(pose, joint_vels)
            self.sock.sendto(msg, self.addr)
            return True
        except socket.error as e:
            logger.error(f"Send failed: {e}")
            return False

    def close(self):
        self.sock.close()
        logger.info("Socket closed")

def main():
    broadcaster = StateBroadcaster(sender_id=3, addr=('192.168.1.255', 9000))
    period = 1.0 / 100
    try:
        while True:
            pose = (0.1, 1.2, 0.0)
            joint_vels = (0.05, 0.0, 0.0)
            broadcaster.send(pose, joint_vels)
            time.sleep(period)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        broadcaster.close()

if __name__ == '__main__':
    main()