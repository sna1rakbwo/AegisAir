#!/usr/bin/env python3
"""SITL ground-station heartbeat; keeps native PX4 preflight checks intact."""
import argparse
import socket
import struct
import time

try:
    from pymavlink import mavutil
except ModuleNotFoundError:
    mavutil = None

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=18572)
parser.add_argument('--duration-s', type=float, default=50.0)
args = parser.parse_args()
def x25(data):
    crc = 0xffff
    for byte in data:
        tmp = byte ^ (crc & 0xff)
        tmp ^= (tmp << 4) & 0xff
        crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xffff
    return crc


def heartbeat_packet(sequence):
    # MAVLink v1 HEARTBEAT (message id 0, CRC extra 50).  This tiny fallback
    # removes an undeclared host dependency while sending the same standard
    # GCS heartbeat as pymavlink.
    payload = struct.pack('<IBBBBB', 0, 6, 8, 0, 4, 3)
    header = bytes((len(payload), sequence & 0xff, 255, 190, 0))
    checksum = x25(header + payload + bytes((50,)))
    return b'\xfe' + header + payload + struct.pack('<H', checksum)


link = mavutil.mavlink_connection(f'udpout:127.0.0.1:{args.port}', source_system=255) if mavutil else None
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if link is None else None
end = time.monotonic() + args.duration_s
sequence = 0
while time.monotonic() < end:
    if link:
        link.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    else:
        sock.sendto(heartbeat_packet(sequence), ('127.0.0.1', args.port))
        sequence += 1
    time.sleep(0.2)
