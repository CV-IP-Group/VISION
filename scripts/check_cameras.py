"""Utility that lists attached video devices via `frame_capture.list_cameras`."""

from __future__ import annotations

import argparse
from typing import Tuple

from frame_capture import list_cameras


def main(max_index: int, timeout: float) -> None:
    devices = list_cameras(max_index=max_index, timeout=timeout)
    if not devices:
        print("No cameras detected. Ensure the device is powered and accessible.")
        return
    print("Detected cameras:")
    for index, description in devices:
        print(f"  - index={index}, description={description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List available cameras for VisionMate capture.")
    parser.add_argument("--max-index", type=int, default=5, help="Maximum camera index to probe.")
    parser.add_argument("--timeout", type=float, default=0.5, help="Per-device capture timeout in seconds.")
    args = parser.parse_args()
    main(args.max_index, args.timeout)
