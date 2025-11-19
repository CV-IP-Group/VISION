#!/usr/bin/env python3
"""Interactive CLI client to verify VisionMate MCP API end-to-end.

Usage:
    python scripts/verify_cli_client.py

The tool posts user-entered text to http://localhost:9000/api/v1/mcp and
pretty-prints the orchestrator response, highlighting intent routing,
vision outputs, adaptive capture resolution, and LiDAR fusion metadata.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

import requests

API_ENDPOINT = "http://localhost:9000/api/v1/mcp"


def _render_objects(objects: List[Dict[str, Any]]) -> str:
    if not objects:
        return "[YOLO] Found: none"
    labels = []
    distances = []
    for item in objects:
        cls = item.get("cls", "?")
        labels.append(cls)
        dist = item.get("dist")
        if dist is None:
            distances.append("--")
        else:
            try:
                distances.append(f"{float(dist):.2f}m")
            except (TypeError, ValueError):
                distances.append(str(dist))
    return f"[YOLO] Found: {', '.join(labels)} | dist: {', '.join(distances)}"


def _render_ocr(result: Dict[str, Any]) -> Optional[str]:
    text = result.get("ocr_text")
    if text:
        return f"[OCR] Text: {text}"
    return None


def _render_meta(meta: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if not isinstance(meta, dict):
        return lines

    capture_resolution = meta.get("capture_resolution") or meta.get("chosen_resolution")
    if capture_resolution:
        lines.append(f"[Meta] capture_resolution: {capture_resolution}")

    lidar_snapshot = meta.get("lidar_snapshot")
    has_lidar = False
    if isinstance(lidar_snapshot, dict):
        # Fresh data might be nested under payload["data"].
        if lidar_snapshot.get("forward_min_distance") is not None:
            has_lidar = True
        elif isinstance(lidar_snapshot.get("data"), dict) and (
            lidar_snapshot["data"].get("forward_min_distance") is not None
        ):
            has_lidar = True
        elif lidar_snapshot.get("fresh"):
            has_lidar = True
    lines.append(f"[Meta] LiDAR Data: {'Present' if has_lidar else 'Missing'}")

    return lines


def _print_response(payload: Dict[str, Any]) -> None:
    intent = payload.get("intent")
    message = payload.get("message")
    data = payload.get("data") or {}
    meta = payload.get("meta") or {}

    print(f"Intent: {json.dumps(intent, ensure_ascii=False)}")
    print(f"Message: {message}")

    objects = data.get("objects") if isinstance(data, dict) else None
    if isinstance(objects, list):
        print(_render_objects(objects))

    ocr_line = _render_ocr(data if isinstance(data, dict) else {})
    if ocr_line:
        print(ocr_line)

    for line in _render_meta(meta):
        print(line)

    # Show any warnings/errors embedded in meta for quick diagnostics.
    error_code = meta.get("error_code")
    if error_code:
        print(f"[Meta] error_code: {error_code}")


def main() -> None:
    print("VisionMate CLI verifier. Press Ctrl+C or enter 'q' to exit.")
    session = requests.Session()

    while True:
        try:
            user_input = input("Enter command (or 'q' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.lower() in {"q", "quit", "exit"}:
            print("Goodbye.")
            break

        if not user_input:
            continue

        payload = {
            "api_version": "v1",
            "text": user_input,
            "camera_index": None,
            "tts": False,
        }

        try:
            response = session.post(API_ENDPOINT, json=payload, timeout=30)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            continue

        print(f"HTTP Status: {response.status_code}")

        try:
            data = response.json()
        except ValueError:
            print("Failed to parse JSON response. Raw content:\n", response.text)
            continue

        _print_response(data)
        print("-" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
