#!/usr/bin/env python3
"""Scan for all nearby BLE devices and identify Revopoint turntables.

Usage:
    python scripts/ble_scan.py           # 10-second scan
    python scripts/ble_scan.py --time 20 # longer scan
    python scripts/ble_scan.py --all     # show every device, not just candidates

The Revopoint turntable advertises service UUID 0xFFE0 (serial-over-BLE).
Even if the device has no readable name, the address printed here can be passed
directly to the scanner:

    python -m scanner turntable-scan --ble-address AA:BB:CC:DD:EE:FF
"""

from __future__ import annotations

import argparse
import asyncio

from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

# UUIDs the Revopoint turntable advertises (some units advertise ffe1, others ffe0)
_REVOPOINT_UUIDS = {
    "0000ffe0-0000-1000-8000-00805f9b34fb",
    "0000ffe1-0000-1000-8000-00805f9b34fb",
}

# Heuristic name fragments (lower-case) — confirmed: "revo_dual_axis_table"
_NAME_HINTS = ("revo_dual", "revo_tt", "revo", "turntable", "tt_")


def _is_candidate(device: BLEDevice, adv: AdvertisementData) -> bool:
    name = (device.name or "").lower()
    if any(h in name for h in _NAME_HINTS):
        return True
    uuids = {str(u).lower() for u in adv.service_uuids}
    if uuids & _REVOPOINT_UUIDS:
        return True
    return False


async def scan(duration: float, show_all: bool) -> None:
    print(f"Scanning for BLE devices ({duration:.0f} s) …\n")

    discovered: dict[str, tuple[BLEDevice, AdvertisementData]] = {}

    def callback(device: BLEDevice, adv: AdvertisementData) -> None:
        discovered[device.address] = (device, adv)

    scanner = BleakScanner(detection_callback=callback)
    await scanner.start()
    await asyncio.sleep(duration)
    await scanner.stop()

    if not discovered:
        print("No BLE devices found. Make sure Bluetooth is enabled.")
        return

    candidates = {
        addr: (d, a) for addr, (d, a) in discovered.items() if _is_candidate(d, a)
    }
    others = {
        addr: (d, a) for addr, (d, a) in discovered.items() if addr not in candidates
    }

    # --- Revopoint candidates ---
    if candidates:
        print("=" * 60)
        print("  REVOPOINT TURNTABLE CANDIDATES")
        print("=" * 60)
        for addr, (dev, adv) in sorted(candidates.items(), key=lambda x: -(x[1][1].rssi or -999)):
            name = dev.name or "(no name)"
            rssi = adv.rssi or "?"
            uuids = ", ".join(str(u) for u in adv.service_uuids) or "none"
            mfr = adv.manufacturer_data
            mfr_str = ", ".join(f"0x{k:04X}: {v.hex()}" for k, v in mfr.items()) if mfr else "none"
            print(f"  Name    : {name}")
            print(f"  Address : {addr}")
            print(f"  RSSI    : {rssi} dBm")
            print(f"  Services: {uuids}")
            print(f"  Mfr data: {mfr_str}")
            print()
        print("  Connect command:")
        first_addr = next(iter(candidates))
        print(f"    python -m scanner turntable-scan --ble-address {first_addr}")
        print()
    else:
        print("No Revopoint turntable candidates found.")
        print("  • Make sure the turntable is powered on.")
        print("  • Disconnect it from the Revo Scan app / iOS app first.")
        print("  • Try scanning longer: python scripts/ble_scan.py --time 20")
        print()

    # --- All other devices (if requested) ---
    if show_all and others:
        print("-" * 60)
        print("  ALL OTHER BLE DEVICES")
        print("-" * 60)
        for addr, (dev, adv) in sorted(others.items(), key=lambda x: -(x[1][1].rssi or -999)):
            name = dev.name or "(no name)"
            rssi = adv.rssi or "?"
            uuids = ", ".join(str(u) for u in adv.service_uuids) or "none"
            print(f"  {name:<30} {addr}  RSSI {rssi:>4}  svcs: {uuids}")
        print()

    print(f"Total devices seen: {len(discovered)}  "
          f"(candidates: {len(candidates)}, other: {len(others)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time", "-t", type=float, default=10.0, metavar="SEC",
                   help="Scan duration in seconds (default 10)")
    p.add_argument("--all", "-a", action="store_true",
                   help="Also list all non-Revopoint BLE devices")
    args = p.parse_args()
    asyncio.run(scan(args.time, args.all))


if __name__ == "__main__":
    main()
