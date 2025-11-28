#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export two depth visualizations for a single image:

1) Raw MiDaS depth (no LiDAR)
2) LiDAR-calibrated depth

Usage examples:

  # 手动给 LiDAR 距离（米）
  python scripts/export_depth_maps.py \
      --image captured_images/test.jpg \
      --lidar-distance 2.5

  # 从 LiDAR 快照 JSON 里读 forward_min_distance
  python scripts/export_depth_maps.py \
      --image captured_images/test.jpg \
      --lidar-json runs/lidar_snapshot.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from vision.backends import RealBackends


def load_lidar_distance(
    lidar_distance: Optional[float],
    lidar_json: Optional[str],
) -> Optional[float]:
    """Resolve LiDAR forward distance in meters."""
    if lidar_distance is not None:
        return float(lidar_distance)

    if lidar_json:
        path = Path(lidar_json)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to read LiDAR JSON: {exc}")
                return None

            cand = payload.get("forward_min_distance")
            try:
                return float(cand) if cand is not None else None
            except (TypeError, ValueError):
                print(f"[WARN] Invalid forward_min_distance in JSON: {cand}")
                return None

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input RGB image path.")
    parser.add_argument(
        "--lidar-distance",
        type=float,
        default=None,
        help="LiDAR forward distance (meters). Overrides JSON.",
    )
    parser.add_argument(
        "--lidar-json",
        type=str,
        default=None,
        help="Optional LiDAR snapshot JSON with 'forward_min_distance'.",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="weights",
        help="Same as main project weights_dir.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs",
        help="Output dir, same as main project run_dir.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save depth PNGs (default: {run_dir}/depth_eval).",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    backend = RealBackends(weights_dir=args.weights_dir, output_dir=args.run_dir)

    print(f"[INFO] Estimating MiDaS depth for {img_path} ...")
    rel_depth = backend.estimate_depth(str(img_path))

    if rel_depth.size == 0:
        raise SystemExit("[ERROR] rel_depth is empty. Check MiDaS setup.")

    # ① 纯视觉：线性投影到 0.4–8m（和主系统保持一致）
    depth_base = backend.linear_depth_projection(rel_depth, near=0.4, far=8.0)

    # ② 带 LiDAR：用 forward_min_distance 做轻度缩放
    lidar_dist = load_lidar_distance(args.lidar_distance, args.lidar_json)
    if lidar_dist is not None:
        print(f"[INFO] Using LiDAR forward distance = {lidar_dist:.2f} m")
        depth_lidar = backend.calibrated_absolute_depth(
            rel_depth,
            lidar_dist=lidar_dist,
        )
    else:
        print("[INFO] No LiDAR distance provided, using base depth for both.")
        depth_lidar = depth_base.copy()

    # 用 backend 的归一化逻辑 + Jet colormap 生成热度图
    depth_base_u8 = backend._normalize_depth_map(depth_base)
    depth_lidar_u8 = backend._normalize_depth_map(depth_lidar)

    heat_base = cv2.applyColorMap(depth_base_u8, cv2.COLORMAP_JET)
    heat_lidar = cv2.applyColorMap(depth_lidar_u8, cv2.COLORMAP_JET)

    # 输出目录
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.run_dir) / "depth_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    raw_path = out_dir / f"{stem}_raw_midas_depth.png"
    lidar_path = out_dir / f"{stem}_lidar_calibrated_depth.png"

    cv2.imwrite(str(raw_path), heat_base)
    cv2.imwrite(str(lidar_path), heat_lidar)

    print(f"[OK] Saved raw MiDaS depth to:    {raw_path}")
    print(f"[OK] Saved LiDAR-calibrated depth to: {lidar_path}")


if __name__ == "__main__":
    main()
