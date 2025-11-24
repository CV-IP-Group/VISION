# import cv2
# import os
# from datetime import datetime
# import time
# import json
#
# # This script includes functions to list available cameras
# # Make a directory to save captured images
# # And Capture images at low, medium, and high resolutions
# # For MCP serve calls
# #
# # Resolutions:
# # low: 640x480 for real-time processing
# # medium: 1280x720 for object recognition
# # high: 1920x1080 for text recognition (OCR)
#
# def list_cameras(max_index=5, timeout=0.5):
#     available_cameras = []
#
#     for i in range(max_index):
#         cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
#         start_time = time.time()
#         if cap.isOpened():
#             ret, frame = cap.read()
#             elapsed = time.time() - start_time
#             if ret and elapsed < timeout:
#                 # I tried to get camera name with some methods but failed
#                 # Maybe use resolution as a proxy is useless
#                 # Since all devices may show the same resolution "640x480" as default
#                 width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#                 height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#                 available_cameras.append((i, f"{int(width)}x{int(height)}"))
#         cap.release()
#
#     return available_cameras
#
# # Run the camera listing function (Not here)
# # camera_list = list_cameras()
#
# # Print the results
# # for index, description in camera_list:
# #   print(f"Camera index {index}: {description}")
#
# def make_save_dir(dir_name="captured_images"):
#     save_dir = dir_name
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir, exist_ok=True)
#     return save_dir
#
# def capture_image(resolution, label, camera_index=1, save_dir="captured_images"):
#     cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
#
#     ret, frame = cap.read()
#     cap.release()
#
#     if ret:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{label}_frame_{timestamp}.jpg"
#         filepath = os.path.join(save_dir, filename)
#         cv2.imwrite(filepath, frame)
#         return filepath
#     else:
#         return None
#
# def capture_low_resolution():
#     return capture_image((640, 480), "low")
#
# def capture_medium_resolution():
#     return capture_image((1280, 720), "medium")
#
# def capture_high_resolution():
#     return capture_image((1920, 1080), "high")
#
# def query_handler(query: dict):
#     action = query.get("action", "")
#     camera_index = query.get("camera_index", 1)
#     save_dir = make_save_dir()
#
#     if action == "call":
#         
#     else:

# --- NEW VERSION ---
"""Cross-platform camera helpers with non-destructive fallbacks."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision.utils import ensure_dir

DEFAULT_CAPTURE_DIR_ENV = "VISIONMATE_CAP_DIR"
DEFAULT_CAMERA_INDEX_ENV = "VISIONMATE_CAMERA_INDEX"

_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "low": (640, 480),
    "medium": (1280, 720),
    "high": (1920, 1080),
}

_FUNCTION_DEFAULTS: Dict[str, str] = {
    "vision.ocr_read": "high",
    "vision.detect_objects": "medium",
    "vision.check_hazards": "low",
    "vision.distance_query": "medium",
    "vision.describe_scene": "medium",
}


def _camera_backend() -> int:
    if os.name == "nt":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def _default_camera_index() -> int:
    try:
        return int(os.getenv(DEFAULT_CAMERA_INDEX_ENV, "0"))
    except ValueError:
        return 0


def make_save_dir(dir_name: Optional[str] = None) -> str:
    base_dir = dir_name or os.getenv(DEFAULT_CAPTURE_DIR_ENV, "captured_images")
    return str(ensure_dir(base_dir))


def list_cameras(max_index: int = 5, timeout: float = 0.5) -> List[Tuple[int, str]]:
    available: List[Tuple[int, str]] = []
    backend = _camera_backend()
    for index in range(max_index):
        cap = cv2.VideoCapture(index, backend)
        start_time = time.time()
        if cap.isOpened():
            ret, frame = cap.read()
            elapsed = time.time() - start_time
            if ret and elapsed < timeout:
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                available.append((index, f"{int(width)}x{int(height)}"))
        cap.release()
    return available


def _capture_with_retries(
    resolution: Tuple[int, int],
    label: str,
    camera_index: Optional[int] = None,
    save_dir: Optional[str] = None,
    retries: int = 2,
    retry_delay: float = 0.05,
) -> Tuple[Optional[str], List[Dict[str, object]]]:
    attempts: List[Dict[str, object]] = []
    idx = _default_camera_index() if camera_index is None else camera_index
    base_dir = make_save_dir(save_dir)
    backend = _camera_backend()

    for attempt in range(retries + 1):
        start = time.time()
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            ret, frame = False, None
            print(f"[INFO] Warming up camera ({label})...")
            for _ in range(15):
                cap.read()
            for read_try in range(retries + 1):
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
                if read_try < retries:
                    time.sleep(retry_delay)
            if ret and frame is not None:
                gray_check = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if np.mean(gray_check) < 5:
                    print("[WARN] Frame is too dark! Retrying capture...")
                    ret = False
                    frame = None
        else:
            ret, frame = False, None
        cap.release()

        elapsed = time.time() - start
        attempts.append(
            {
                "attempt": attempt + 1,
                "elapsed": elapsed,
                "success": bool(ret),
                "resolution": label,
            }
        )

        if ret and frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{label}_frame_{timestamp}.jpg"
            filepath = os.path.join(base_dir, filename)
            cv2.imwrite(filepath, frame)
            latest_root = ensure_dir("runs")
            latest_rgb_path = latest_root / "latest_rgb.jpg"
            cv2.imwrite(str(latest_rgb_path), frame)
            return filepath, attempts

        if attempt < retries:
            time.sleep(retry_delay)

    return None, attempts


def capture_image(
    resolution: Tuple[int, int],
    label: str,
    camera_index: Optional[int] = None,
    save_dir: Optional[str] = None,
    retries: int = 2,
) -> Optional[str]:
    path, _ = _capture_with_retries(
        resolution,
        label,
        camera_index=camera_index,
        save_dir=save_dir,
        retries=retries,
    )
    return path


def capture_low_resolution() -> Optional[str]:
    return capture_image(_RESOLUTIONS["low"], "low")


def capture_medium_resolution() -> Optional[str]:
    return capture_image(_RESOLUTIONS["medium"], "medium")


def capture_high_resolution() -> Optional[str]:
    return capture_image(_RESOLUTIONS["high"], "high")


def query_handler(query: Dict[str, object]) -> Dict[str, object]:
    action = str(query.get("action") or "call").lower()
    if action == "list":
        max_index = query.get("max_index")
        try:
            max_index_int = int(max_index) if max_index is not None else 5
        except (TypeError, ValueError):
            max_index_int = 5
        return {
            "status": "ok",
            "message": "Available camera list returned.",
            "cameras": list_cameras(max_index=max_index_int),
        }

    if action not in {"call", "capture"}:
        return {
            "status": "warn",
            "message": "Unknown action type ignored.",
        }

    desired = str(query.get("resolution") or "").lower()
    function = query.get("function")
    if not desired and isinstance(function, str):
        desired = _FUNCTION_DEFAULTS.get(function, "")
    if desired not in _RESOLUTIONS:
        desired = "medium"

    overrides = [desired, "high", "medium", "low"]
    resolutions = []
    for key in overrides:
        if key in _RESOLUTIONS and key not in resolutions:
            resolutions.append(key)

    camera_index = query.get("camera_index")
    camera_index = int(camera_index) if isinstance(camera_index, (int, str)) and str(camera_index).isdigit() else None

    directory = query.get("directory") or query.get("save_dir") or os.getenv(DEFAULT_CAPTURE_DIR_ENV)

    attempts_all: List[Dict[str, object]] = []
    for key in resolutions:
        path, attempts = _capture_with_retries(
            _RESOLUTIONS[key],
            key,
            camera_index=camera_index,
            save_dir=directory,
        )
        attempts_all.extend(attempts)
        if path:
            return {
                "status": "ok",
                "message": "Frame captured successfully.",
                "path": path,
                "resolution": key,
                "attempts": attempts_all,
            }

    return {
        "status": "warn",
        "message": "Camera unavailable after exhausting fallback resolutions.",
        "path": None,
        "resolution": resolutions[0] if resolutions else None,
        "attempts": attempts_all,
    }

