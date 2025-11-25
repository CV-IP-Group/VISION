"""Quick visualization utility for the latest VisionMate run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RUN_DIR = Path("runs")
STATUS_PATH = RUN_DIR / "latest_status.json"
LATEST_DEPTH_FALLBACK = RUN_DIR / "latest_depth.jpg"

# --- 1. 字体设置 (解决中文显示为方块的问题) ---
# Matplotlib 会按顺序尝试寻找可用字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


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
    # BGR 转 RGB，用于 matplotlib 显示
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _get_hazard_color(level: str) -> str:
    """根据危险等级返回颜色 HEX 值 (美化版)"""
    level = str(level).lower()
    if "high" in level or "danger" in level:
        return "#D32F2F"  # 深红
    elif "medium" in level or "warn" in level:
        return "#F57C00"  # 橙色
    elif "safe" in level or "low" in level:
        return "#388E3C"  # 绿色
    return "#333333"      # 黑色/未知


def main() -> None:
    try:
        data = _load_latest_status()
    except Exception as e:
        print(f"Error loading status: {e}")
        return

    # 加载图像
    img_orig = _imread_or_none(data.get("image_path"))
    img_yolo = _imread_or_none(data.get("yolo_debug_path"))
    img_depth = _imread_or_none(data.get("depth_overlay_path"))
    
    # 深度图回退机制
    if img_depth is None:
        img_depth = _imread_or_none(str(LATEST_DEPTH_FALLBACK))

    # 准备要显示的图像列表
    valid_images: List[Tuple[str, Any]] = []
    if img_orig is not None:
        valid_images.append(("原始图像 (Original)", img_orig))
    if img_yolo is not None:
        valid_images.append(("YOLO 检测 (Detection)", img_yolo))
    if img_depth is not None:
        valid_images.append(("深度估测 (Depth)", img_depth))

    if not valid_images:
        print("没有可用的图像，仅打印 JSON 数据：")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    num_imgs = len(valid_images)
    
    # --- 布局核心 (GridSpec) ---
    # 设置背景色为浅灰，类似 Dashboard 风格
    fig = plt.figure(figsize=(5 * num_imgs, 7), dpi=100)
    fig.patch.set_facecolor('#f0f0f0') 
    
    # height_ratios=[4, 1.2] 确保图片区域大，文字区域适中
    gs = GridSpec(2, num_imgs, height_ratios=[4, 1.2], figure=fig)

    # 1. 绘制图片区域
    for idx, (title, img) in enumerate(valid_images):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(img)
        # 标题加粗，字体稍大
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.axis("off")
        
        # 可选：给每个图片加一个细微的边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#cccccc')

    # 2. 绘制文字区域 (底部通栏)
    ax_text = fig.add_subplot(gs[1, :]) # 占用第2行的所有列
    ax_text.axis("off")
    
    # 准备数据
    hazard_level = str(data.get('hazard_level', 'Unknown'))
    hazard_color = _get_hazard_color(hazard_level)
    
    # 文本信息 (去掉了 Emoji，避免乱码；增加了中文标签)
    # 注意：这里不再强制 family="monospace"，这解决了中文方块问题
    info_text = (
        f"【意图】: {data.get('intent', 'N/A')}\n"
        f"【输入】: {data.get('raw_text', 'N/A')}\n"
        f"【消息】: {data.get('message', 'N/A')}"
    )

    # 绘制左侧主要文本
    ax_text.text(
        0.02, 0.8, 
        info_text, 
        fontsize=11, 
        va="top", 
        ha="left", 
        linespacing=1.8,
        color="#333333"
    )

    # 绘制右侧 Hazard Level (带圆角背景框)
    bbox_props = dict(boxstyle="round,pad=0.6", fc=hazard_color, ec="none", alpha=0.9)
    
    ax_text.text(
        0.98, 0.8, 
        f"Hazard Level: {hazard_level.upper()}", 
        fontsize=14, 
        fontweight='bold',
        color="white", # 深色背景配白色文字
        va="top", 
        ha="right",
        bbox=bbox_props
    )

    # 添加分割线 (图片和文字之间)
    line = plt.Line2D([0.02, 0.98], [0.95, 0.95], 
                      transform=ax_text.transAxes, color="#cccccc", linewidth=1, linestyle="-")
    ax_text.add_line(line)

    plt.tight_layout()
    print(f"显示可视化结果，包含 {num_imgs} 张图片...")
    plt.show()


if __name__ == "__main__":
    main()