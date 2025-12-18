"""
FL modules

If already store FL weights at PC,
can distribute them back directly to the devices.
"""

import asyncio
import re
import numpy as np
from bleak import BleakClient, BleakScanner


DEVICES = {
    "user1": "Sender_1",   # or MAC address (for MacOS user)
    "user2": "Sender_2",
}

# stored model weights.h you want to distribute back to devices
WEIGHTS_PATH = "ble_weights/global_fedavg_round10.h"

SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
WRITE_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"

SESSION_ID = 10                 # session_id sent to MCU
MAX_CHUNK_PAYLOAD = 180
HEADER_SIZE = 11
FLAG_LAST_CHUNK = 0x01


# =========================================================
# read weights from `.h`
# =========================================================


def load_weights_from_h(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # read WEIGHTS_CNT
    m_cnt = re.search(r"#define\s+WEIGHTS_CNT\s+(\d+)", text)
    if not m_cnt:
        raise RuntimeError("WEIGHTS_CNT not found in header")
    expected_cnt = int(m_cnt.group(1))

    # extract weights in {...}
    m_body = re.search(r"\{([\s\S]*?)\}", text)
    if not m_body:
        raise RuntimeError("Weight array body not found")

    body = m_body.group(1)

    # remove 'f' at end from float
    nums = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", body)
    weights = np.array(nums, dtype=np.float32)

    if weights.size != expected_cnt:
        raise RuntimeError(
            f"WEIGHTS_CNT mismatch: header={expected_cnt}, parsed={weights.size}"
        )

    print(f"✅ Loaded {weights.size} weights from {path}")
    print(f"   First 3: {weights[:3]}")
    print(f"   Last  3: {weights[-3:]}")

    return weights


# =========================================================
# BLE
# =========================================================


async def resolve_identifier(identifier: str) -> str:
    if "-" in identifier:
        return identifier

    print(f"🔍 Scanning for {identifier} ...")
    devices = await BleakScanner.discover(timeout=5.0)
    for d in devices:
        if d.name == identifier:
            print(f"✅ Found {identifier}: {d.address}")
            return d.address

    raise RuntimeError(f"Device {identifier} not found")


def pack_weights(weights: np.ndarray, session_id: int):
    data = weights.astype(np.float32).tobytes()
    total_size = len(data)
    total_chunks = (total_size + MAX_CHUNK_PAYLOAD - 1) // MAX_CHUNK_PAYLOAD

    packets = []
    offset = 0

    print(f"📦 Packing: {total_size} bytes → {total_chunks} chunks")

    for chunk_id in range(total_chunks):
        payload = data[offset : offset + MAX_CHUNK_PAYLOAD]
        offset += len(payload)

        is_last = (chunk_id == total_chunks - 1)
        flags = FLAG_LAST_CHUNK if is_last else 0

        header = bytes([
            (session_id >> 8) & 0xFF,
            session_id & 0xFF,
            (chunk_id >> 8) & 0xFF,
            chunk_id & 0xFF,
            (total_chunks >> 8) & 0xFF,
            total_chunks & 0xFF,
            (total_size >> 24) & 0xFF,
            (total_size >> 16) & 0xFF,
            (total_size >> 8) & 0xFF,
            total_size & 0xFF,
            flags,
        ])

        packets.append(header + payload)

    return packets


# =========================================================
# distribute to MCU
# =========================================================


async def push_to_mcu(user_id: str, identifier: str, packets):
    try:
        addr = await resolve_identifier(identifier)
        print(f"🔗 [{user_id}] Connecting to {addr}")

        async with BleakClient(addr) as client:
            print(f"✅ [{user_id}] Connected")

            for i, pkt in enumerate(packets):
                await client.write_gatt_char(WRITE_UUID, pkt, response=True)
                await asyncio.sleep(0.02)

                if (i + 1) % 10 == 0 or i == len(packets) - 1:
                    print(f"📤 [{user_id}] {i+1}/{len(packets)}")

            print(f"🎉 [{user_id}] Weights sent successfully")

    except Exception as e:
        print(f"❌ [{user_id}] Failed: {e}")


# =========================================================
# main
# =========================================================


async def main():

    weights = load_weights_from_h(WEIGHTS_PATH)

    packets = pack_weights(weights, SESSION_ID)

    tasks = [
        push_to_mcu(user_id, identifier, packets)
        for user_id, identifier in DEVICES.items()
    ]

    await asyncio.gather(*tasks)

    print("\n🌍 Broadcast finished for all MCUs")

if __name__ == "__main__":
    asyncio.run(main())
