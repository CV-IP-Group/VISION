"""CLI that fires a series of text commands at the Orchestrator and prints the payloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import cv2

from orchestrator.router import Orchestrator
from vision.types import VisionResult

DEFAULT_COMMANDS = [
    "检测前方障碍",
    "前方有什么物体",
    "离我有多远",
    "找找目标",
    "读一下文字",
]


def ensure_lidar_snapshot(path: Path, distance: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"forward_min_distance": distance}
    path.write_text(json.dumps(payload), encoding="utf-8")


def serialize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, VisionResult):
            safe[key] = value.to_dict()
        else:
            try:
                json.dumps(value)
                safe[key] = value
            except (TypeError, ValueError):
                safe[key] = str(value)
    return safe


def main(
    capture_dir: Path,
    run_dir: Path,
    weights_dir: Path,
    lidar_snapshot: Path,
    commands: List[str],
    camera_index: int | None,
    sample_capture: bool,
) -> None:
    os.environ.setdefault("VISIONMATE_CAPTURE_DIR", str(capture_dir))
    os.environ.setdefault("VISIONMATE_RUN_DIR", str(run_dir))
    os.environ.setdefault("VISIONMATE_WEIGHTS_DIR", str(weights_dir))
    os.environ.setdefault("VISIONMATE_LIDAR_SNAPSHOT", str(lidar_snapshot))
    os.environ.setdefault("VISIONMATE_LIDAR_TTL", "5")
    if camera_index is not None:
        os.environ.setdefault("VISIONMATE_CAMERA_INDEX", str(camera_index))

    ensure_lidar_snapshot(lidar_snapshot, distance=0.5)

    if sample_capture:
        import frame_capture

        sample_path = capture_dir / "sample_frame.jpg"
        sample_path.parent.mkdir(parents=True, exist_ok=True)

        def _stub_capture(resolution: tuple[int, int], label: str, *_: Any, **__: Any) -> str:
            dummy = np.zeros((resolution[1], resolution[0], 3), dtype="uint8")
            cv2.imwrite(str(sample_path), dummy)
            return str(sample_path)

        frame_capture.capture_image = _stub_capture

    orchestrator = Orchestrator(
        capture_dir=str(capture_dir),
        run_dir=str(run_dir),
        weights_dir=str(weights_dir),
    )

    for text in commands:
        payload = orchestrator.handle_text(text, camera_index=camera_index)
        safe_payload = serialize_payload(payload)
        print("\n--- COMMAND ---")
        print(text)
        print(json.dumps(safe_payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send keyword commands to the Orchestrator and dump the payloads.")
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path("captures"),
        help="Where capture images will be stored/loaded.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs"),
        help="Working directory for run artifacts.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Directory that contains model weights.",
    )
    parser.add_argument(
        "--lidar-snapshot",
        type=Path,
        default=Path("data/fake_lidar_snapshot.json"),
        help="JSON file that represents the LiDAR snapshot for fusion.",
    )
    parser.add_argument(
        "--commands",
        nargs="*",
        default=DEFAULT_COMMANDS,
        help="Text commands to send to the orchestrator. Defaults to the built-in keywords.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Optional camera index to force via environment.",
    )
    parser.add_argument(
        "--sample-capture",
        action="store_true",
        help="Use a synthetic capture frame instead of talking to the camera.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        capture_dir=args.capture_dir,
        run_dir=args.run_dir,
        weights_dir=args.weights_dir,
        lidar_snapshot=args.lidar_snapshot,
        commands=args.commands,
        camera_index=args.camera_index,
        sample_capture=args.sample_capture,
    )
