"""Simple utility to keep the camera open and mirror the feed on the screen."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import cv2


def _camera_backend() -> int:
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a persistent camera preview window and keep the feed alive."
    )
    parser.add_argument("-i", "--camera", type=int, default=int(os.getenv("VISIONMATE_CAMERA_INDEX", "0")), help="Camera index to open.")
    parser.add_argument("-r", "--resolution", choices=["low", "medium", "high"], default="medium", help="Capture resolution used for the preview.")
    parser.add_argument("-w", "--window", default="VisionMate Camera", help="Window title for the preview.")
    parser.add_argument("-n", "--no-overlay", action="store_true", help="Disable timestamp overlay." )
    parser.add_argument("--fps", type=float, default=30.0, help="Cap the preview frame rate to keep CPU usage manageable.")
    return parser.parse_args()


def _resolution_to_size(label: str) -> tuple[int, int]:
    mapping = {
        "low": (640, 480),
        "medium": (1280, 720),
        "high": (1920, 1080),
    }
    return mapping.get(label, (1280, 720))


def _draw_overlay(frame, overlay_text: str) -> None:
    cv2.putText(
        frame,
        overlay_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def _handle_interrupt(signum, frame):
    raise KeyboardInterrupt


def main() -> None:
    args = _parse_args()
    resolution = _resolution_to_size(args.resolution)
    backend = _camera_backend()

    signal.signal(signal.SIGINT, _handle_interrupt)

    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        print(f"无法打开摄像头 {args.camera}。")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    target_delay = max(1, int(1000 / max(1.0, args.fps)))

    print("按下 'q' 退出预览，摄像头将保持打开状态。")

    try:
        while True:
            started = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("摄像头帧读取失败，正在尝试重新连接...", end="\r")
                time.sleep(0.5)
                continue

            if not args.no_overlay:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                overlay_text = f"Camera {args.camera} · {timestamp}"
                _draw_overlay(frame, overlay_text)

            cv2.imshow(args.window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elapsed = int((time.time() - started) * 1000)
            if elapsed < target_delay:
                time.sleep((target_delay - elapsed) / 1000)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头预览已停止。")


if __name__ == "__main__":
    main()
