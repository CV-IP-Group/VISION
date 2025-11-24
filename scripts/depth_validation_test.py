import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.backends import StubBackends
from vision.engine import CVModuleProcess


def pick_image() -> Path:
    base = Path("captured_images")
    if not base.exists():
        raise FileNotFoundError("captured_images directory missing")
    for child in sorted(base.iterdir()):
        if child.is_file():
            return child
    raise FileNotFoundError("no captured image found")


def run_test():
    image_path = pick_image()
    print(f"Using image for sizing: {image_path}")
    img = Image.open(image_path)
    h, w = img.height, img.width
    cv = CVModuleProcess(backends=StubBackends())

    summary = []

    # TEST1
    depth = np.ones((h, w), dtype=np.float32) * 3.5
    valid = cv._is_valid_depth(depth, (h, w))
    resolved = cv._resolve_depth_map(str(image_path), None, depth, None)
    passed = valid and resolved is not None
    print(f"TEST1 valid depth → {'PASS' if passed else 'FAIL'}")
    if resolved is not None:
        print(f"  resolved shape: {resolved.shape}")
        print(f"  min/max: {float(np.min(resolved)):.2f}/{float(np.max(resolved)):.2f}")
    summary.append(f"TEST1: {'pass' if passed else 'fail'} depth valid check")

    # TEST2
    shape_bad = np.zeros((max(1, h // 2), max(1, w // 2)), dtype=np.float32)
    valid_shape = cv._is_valid_depth(shape_bad, (h, w))
    resolved_shape = cv._resolve_depth_map(str(image_path), shape_bad, None, None)
    passed = not valid_shape and resolved_shape is None
    print(f"TEST2 shape mismatch → {'PASS' if passed else 'FAIL'}")
    summary.append(f"TEST2: {'pass' if passed else 'fail'} shape mismatch")

    # TEST3
    nan_depth = np.ones((h, w), dtype=np.float32)
    nan_depth[10 % h, 10 % w] = np.nan
    valid_nan = cv._is_valid_depth(nan_depth, (h, w))
    resolved_nan = cv._resolve_depth_map(str(image_path), nan_depth, None, None)
    passed = not valid_nan and resolved_nan is None
    print(f"TEST3 nan → {'PASS' if passed else 'FAIL'}")
    summary.append(f"TEST3: {'pass' if passed else 'fail'} contains nan")

    # TEST4
    neg_depth = np.ones((h, w), dtype=np.float32)
    neg_depth[0, 0] = -5.0
    valid_neg = cv._is_valid_depth(neg_depth, (h, w))
    resolved_neg = cv._resolve_depth_map(str(image_path), neg_depth, None, None)
    passed = not valid_neg and resolved_neg is None
    print(f"TEST4 negative → {'PASS' if passed else 'FAIL'}")
    summary.append(f"TEST4: {'pass' if passed else 'fail'} negative value")

    print("[Depth Validation Test Summary]")
    for line in summary:
        print(line)


if __name__ == "__main__":
    run_test()
