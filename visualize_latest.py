"""Quick visualization utility for the latest VisionMate run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import matplotlib.pyplot as plt

RUN_DIR = Path("runs")
STATUS_PATH = RUN_DIR / "latest_status.json"
LATEST_DEPTH_FALLBACK = RUN_DIR / "latest_depth.jpg"


def _load_latest_status() -> Dict[str, Any]:
    if not STATUS_PATH.exists():
        raise FileNotFoundError(f"{STATUS_PATH} 不存在，请先运行一次指令。")
    with STATUS_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _imread_or_none(path_str: Optional[str]) -> Optional[Any]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    image = cv2.imread(str(path))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> None:
    data = _load_latest_status()
    img_orig = _imread_or_none(data.get("image_path"))
    img_yolo = _imread_or_none(data.get("yolo_debug_path"))
    img_depth = _imread_or_none(data.get("depth_overlay_path"))
    if img_depth is None:
        img_depth = _imread_or_none(str(LATEST_DEPTH_FALLBACK))

    text_lines = [
        f"Intent: {data.get('intent')}",
        f"Raw text: {data.get('raw_text')}",
        f"Hazard level: {data.get('hazard_level')}",
        f"Message: {data.get('message')}",
    ]

    images = [img for img in (img_orig, img_yolo, img_depth) if img is not None]
    if not images:
        print("没有可用的图像，只打印 latest_status.json：")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    fig_cols = len(images)
    fig, axes = plt.subplots(2, fig_cols, figsize=(5 * fig_cols, 8))

    col_idx = 0
    titles = ["原始图像", "YOLO 带框图", "MiDaS 深度叠加"]
    sources = [img_orig, img_yolo, img_depth]
    for title, img in zip(titles, sources):
        if img is None:
            continue
        axes[0][col_idx].imshow(img)
        axes[0][col_idx].set_title(title)
        axes[0][col_idx].axis("off")
        col_idx += 1

    axes[1][0].axis("off")
    axes[1][0].text(
        0.05,
        0.95,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=11,
        wrap=True,
    )
    for idx in range(1, fig_cols):
        axes[1][idx].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
