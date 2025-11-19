"""WARNING: THIS GENERATES FAKE LiDAR DATA FOR DEVELOPMENT ONLY.
Generate a simple LiDAR snapshot JSON so Orchestrator always has a fresh fake reading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a LiDAR snapshot JSON file with a simulated forward distance.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/fake_lidar_snapshot.json"),
        help="Path where the snapshot will be written.",
    )
    parser.add_argument(
        "-d",
        "--distance",
        type=float,
        default=0.5,
        help="Approximate forward_min_distance to encode (meters).",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"forward_min_distance": args.distance, "source": "mock"}
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote LiDAR snapshot to {args.output.resolve()} with forward_min_distance={args.distance}")


if __name__ == "__main__":
    main()
