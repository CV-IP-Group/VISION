"""Placeholder and real backends for computer vision tasks.

This module keeps the original ``StubBackends`` for tests and adds a minimal
``RealBackends`` implementation that wires together YOLOv8n, EasyOCR and a
MiDaS-small depth estimator. The goal is to provide a concrete yet lightweight
backend without changing the public CV engine API.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from ultralytics import YOLO
import easyocr
import timm

from .types import Box
from .utils import ensure_dir


class StubBackends:
    """Lightweight stand-ins for actual CV algorithms (kept for tests)."""

    def detect_objects(self, image_path: str, **_: Dict) -> List[Box]:  # pragma: no cover - placeholder
        return []

    def check_hazards(self, image_path: str, **_: Dict) -> Dict[str, str]:  # pragma: no cover - placeholder
        return {
            "status": "clear",
            "message": "未检测到明显障碍（占位结果）。",
        }

    def distance_query(self, image_path: str, **_: Dict) -> Dict[str, Optional[float]]:  # pragma: no cover - placeholder
        return {
            "distance_m": None,
            "message": "距离估计占位，等待深度模型集成。",
            "depth_path": None,
            "overlay_path": None,
        }

    def describe_scene(self, image_path: str, **_: Dict) -> str:  # pragma: no cover - placeholder
        return "场景描述模块尚未接入真实模型。"

    def estimate_depth(self, image_path: str, **_: Dict) -> tuple[np.ndarray, None, None]:  # pragma: no cover - placeholder
        return np.zeros((1, 1), dtype="float32"), None, None

    def ocr_read(self, image_path: str, **_: Dict) -> str:  # pragma: no cover - placeholder
        return "OCR 功能尚未启用，稍后再试。"

    @property
    def depth_error(self) -> Optional[str]:
        return None

    def calibrated_absolute_depth(
        self,
        rel_depth: Optional[np.ndarray],
        lidar_dist: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        return rel_depth


@dataclass
class _DetBox:
    """Internal helper representation before mapping to :class:`Box`."""

    xyxy: Tuple[int, int, int, int]
    cls: str
    score: float
    dist: Optional[float] = None


class RealBackends:
    """Minimal real CV backend using YOLOv8n + EasyOCR + MiDaS-small.

    Assumptions
    ----------
        * YOLOv8n weights are manually downloaded to ``weights/yolov8n.pt``.
        * EasyOCR is installed and configured for simplified Chinese and English.
        * MiDaS-small is loaded via ``timm`` and used for relative depth; a simple
            normalization heuristic converts it into an approximate metric distance
            for demo purposes.
        * You can override which MiDaS variant to load (e.g., ``midas_v21``) via
            ``VISIONMATE_MIDAS_MODEL`` or point to local weights via
            ``VISIONMATE_MIDAS_WEIGHTS`` / ``weights/{model}.pt``.
    """

    def __init__(
        self,
        weights_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        device: Optional[str] = None,
        midas_model: str | None = None,
        midas_weights: str | Path | None = None,
    ) -> None:
        weights_root = Path(weights_dir or "weights")
        self._output_dir = ensure_dir(Path(output_dir or "runs"))
        self._depth_dir = ensure_dir(self._output_dir / "depth")
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._depth_error: Optional[str] = None
        self._midas_model = self._determine_midas_model(midas_model, weights_root)
        self._midas_weights_path = self._resolve_midas_weights_path(
            midas_weights,
            weights_root,
            self._midas_model,
        )
        self._midas_transform_requires_numpy = False

        # YOLOv8n
        yolo_path = weights_root / "yolov8n.pt"
        if not yolo_path.exists():  # 明确失败信息，方便课堂调试
            raise FileNotFoundError(f"YOLOv8n 权重文件未找到: {yolo_path}")
        self._yolo = YOLO(str(yolo_path))

        # EasyOCR (中/英)
        self._ocr = easyocr.Reader(["ch_sim", "en"], gpu=torch.cuda.is_available())

        self._midas, self._midas_transform = self._setup_midas()

    def _resolve_midas_weights_path(
        self,
        explicit: str | Path | None,
        weights_root: Path,
        model_name: str,
    ) -> Path | None:
        """Find an override path for MiDaS weights if provided."""

        if explicit is not None:
            explicit_path = Path(explicit)
            if not explicit_path.exists():
                raise FileNotFoundError(f"MiDaS 权重文件未找到: {explicit_path}")
            return explicit_path

        env_path = os.getenv("VISIONMATE_MIDAS_WEIGHTS")
        if env_path:
            env_candidate = Path(env_path)
            if not env_candidate.exists():
                raise FileNotFoundError(f"MiDaS 权重文件未找到: {env_candidate}")
            return env_candidate

        fallback = weights_root / f"{model_name}.pt"
        if fallback.exists():
            return fallback
        return None

    def _determine_midas_model(self, explicit_model: str | None, weights_root: Path) -> str:
        """Resolve which MiDaS model name should be used."""

        candidates = [explicit_model, os.getenv("VISIONMATE_MIDAS_MODEL")]
        for candidate in candidates:
            if candidate and candidate.strip():
                return candidate.strip()

        discovered = self._discover_midas_weight_model(weights_root)
        if discovered:
            return discovered

        return "MiDaS_small"

    def _discover_midas_weight_model(self, weights_root: Path) -> str | None:
        """Detect any local MiDaS weight files (e.g. `midas_v21_small_256.pt`)."""

        for path in sorted(weights_root.glob("midas_*.pt")):
            if path.is_file():
                return path.stem
        return None

    def _setup_midas(self) -> tuple[torch.nn.Module, Any]:
        """Attempt to create a MiDaS model via local weights, torch.hub, or timm."""

        if self._midas_weights_path is not None:
            return self._load_local_midas()

        return self._load_from_torch_hub(self._midas_model)

    def _load_local_midas(self) -> tuple[torch.nn.Module, Any]:
        """Load a MiDaS model from a provided checkpoint."""

        assert self._midas_weights_path is not None
        if not self._midas_weights_path.exists():
            raise FileNotFoundError(f"MiDaS 权重文件未找到: {self._midas_weights_path}")
        print(f"[RealBackends] 使用本地 MiDaS 权重: {self._midas_weights_path}")
        model = timm.create_model(self._midas_model, pretrained=False)
        state = torch.load(self._midas_weights_path, map_location="cpu")
        if isinstance(state, dict):
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state_dict = state["state_dict"]
            elif "model" in state and isinstance(state["model"], dict):
                state_dict = state["model"]
            else:
                state_dict = state
        else:
            state_dict = state
        model.load_state_dict(state_dict)
        model.eval().to(self._device)
        return model, self._torchvision_midas_transform()

    def _load_from_torch_hub(self, model_name: str) -> tuple[torch.nn.Module, Any]:
        """Download MiDaS via torch.hub and obtain its predefined transforms."""

        model = torch.hub.load("intel-isl/MiDaS", model_name)
        model.eval().to(self._device)
        transforms_module = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = self._resolve_hub_transform(transforms_module, model_name)
        print(f"[RealBackends] 使用 torch.hub 下载 MiDaS({model_name})")
        return model, transform

    def _load_from_timm(self, model_name: str) -> tuple[torch.nn.Module, Any]:
        """Fallback to timm-provided MiDaS implementation."""

        model = timm.create_model(model_name, pretrained=True)
        model.eval().to(self._device)
        return model, self._torchvision_midas_transform()

    def _resolve_hub_transform(self, transforms_module: Any, model_name: str) -> Any:
        """Pick the transform best suited for the downloaded MiDaS variant."""

        if "small" in model_name and hasattr(transforms_module, "small_transform"):
            return transforms_module.small_transform
        if hasattr(transforms_module, "dpt_transform"):
            return transforms_module.dpt_transform
        if hasattr(transforms_module, "default_transform"):
            return transforms_module.default_transform
        return self._torchvision_midas_transform()

    def _apply_midas_transform(self, img: Image.Image) -> torch.Tensor:
        """Handle MiDaS transforms that want numpy inputs or dict outputs."""

        def _array(image: Image.Image) -> np.ndarray:
            return np.asarray(image, dtype="float32")

        target = img if not self._midas_transform_requires_numpy else _array(img)
        try:
            result = self._midas_transform(target)
        except TypeError:
            if self._midas_transform_requires_numpy:
                raise
            self._midas_transform_requires_numpy = True
            target = _array(img)
            result = self._midas_transform(target)

        if isinstance(result, dict):
            result = result.get("image", result)

        if not isinstance(result, torch.Tensor):
            raise TypeError("MiDaS transform must return a tensor-like object.")

        return result

    def _torchvision_midas_transform(self) -> Any:
        return T.Compose(
            [
                T.Resize(256),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    # ------------------------------------------------------------------
    # 公共接口：与 CVModuleProcess 期望匹配
    # ------------------------------------------------------------------
    def detect_objects(self, image_path: str, **_: Dict) -> List[Box]:
        """Run YOLOv8n on an image and return :class:`Box` list.

        Distance estimation is *not* injected here; it is left to the CV engine
        so that depth can be computed once and reused across tasks.
        """

        img = Image.open(image_path).convert("RGB")
        results = self._yolo(img, verbose=False)[0]
        names = results.names

        boxes: List[Box] = []
        for det in results.boxes:
            x1, y1, x2, y2 = det.xyxy[0].tolist()
            cls_id = int(det.cls[0].item())
            score = float(det.conf[0].item())
            cls_name = names.get(cls_id, str(cls_id))
            boxes.append(
                Box(
                    xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    cls=cls_name,
                    score=score,
                    dist=None,
                )
            )
        return boxes

    def ocr_read(self, image_path: str, **_: Dict) -> str:
        """Run EasyOCR on the image and return concatenated text result."""

        img = cv2.imread(image_path)
        if img is None:
            return ""
        results = self._ocr.readtext(img)
        texts: List[str] = []
        for _box, text, conf in results:
            if not text:
                continue
            texts.append(text.strip())
        return " ".join(texts)

    # 其余占位接口保持简单实现，以便与现有 CVModuleProcess 兼容
    def check_hazards(self, image_path: str, **_: Dict) -> Dict[str, str]:  # pragma: no cover - simple rule placeholder
        return {
            "status": "unknown",
            "message": "危险检测逻辑将在 CVModuleProcess 中基于深度/雷达实现。",
        }

    def distance_query(self, image_path: str, **_: Dict) -> Dict[str, Optional[float]]:  # pragma: no cover - placeholder
        rel_depth = self.estimate_depth(image_path)
        abs_depth = self.calibrated_absolute_depth(rel_depth)
        approx = None
        if abs_depth.size:
            median_value = float(np.nanmedian(abs_depth))
            approx = None if np.isnan(median_value) else median_value
        depth_path, overlay_path = self._write_depth_artifacts(image_path, abs_depth)
        return {
            "distance_m": approx,
            "message": "已生成 MiDaS 深度图及叠加图，结果保存在运行目录。",
            "depth_path": depth_path,
            "overlay_path": overlay_path,
        }

    def describe_scene(self, image_path: str, **_: Dict) -> str:  # pragma: no cover - placeholder
        return "场景描述将由上层根据检测结果与深度图组合生成。"

    # ------------------------------------------------------------------
    # 深度估计相关
    # ------------------------------------------------------------------
    def estimate_depth(self, image_path: str) -> np.ndarray:
        """Estimate relative depth map using MiDaS-small.

        Returns a HxW float32 numpy array; larger values correspond to closer
        regions in the scene.
        """
    # 如果当前环境 timm 不支持 MiDaS_small，则返回一个 1x1 的占位深度图
        if self._midas is None or self._midas_transform is None:
            return np.zeros((1, 1), dtype="float32")

        img = Image.open(image_path).convert("RGB")
        transformed = self._apply_midas_transform(img)
        if transformed.ndim == 4:
            inp = transformed.to(self._device)
        else:
            inp = transformed.unsqueeze(0).to(self._device)
        with torch.no_grad():
            depth = self._midas(inp)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        depth = F.interpolate(
            depth,
            size=img.size[::-1],  # (H, W)
            mode="bicubic",
            align_corners=False,
        )
        return depth[0, 0].cpu().numpy().astype("float32")

    def calibrated_absolute_depth(
        self,
        rel_depth: Optional[np.ndarray],
        lidar_dist: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        if rel_depth is None or rel_depth.size == 0:
            return rel_depth
        d = rel_depth
        try:
            d_max = float(np.nanpercentile(d, 95))
        except ValueError:
            return self.linear_depth_projection(d)
        if d_max <= 0.0:
            return self.linear_depth_projection(d)
        scale = (lidar_dist if lidar_dist and lidar_dist > 0 else 1.0) * d_max
        depth_map = scale / (d + 1e-6)
        depth_map = np.clip(depth_map, 0.1, 20.0)
        return depth_map.astype("float32")

    @staticmethod
    def linear_depth_projection(rel_depth: np.ndarray, near: float = 0.5, far: float = 8.0) -> np.ndarray:
        """Convert relative depth to a crude metric distance estimate.

        This uses percentile-based normalization and a linear mapping. The
        result is only meant for demo / heuristic fusion and **not** for
        safety-critical use.
        """

        if rel_depth.size == 0:
            return rel_depth
        d = rel_depth
        d_min = float(np.percentile(d, 5))
        d_max = float(np.percentile(d, 95))
        if d_max <= d_min:
            return np.full_like(d, far, dtype="float32")
        rel = np.clip((d - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)
        d_abs = near + (1.0 - rel) * (far - near)
        return d_abs.astype("float32")

    relative_to_absolute_depth = linear_depth_projection

    @staticmethod
    def attach_depth_to_boxes(boxes: List[Box], depth_map: np.ndarray) -> List[Box]:
        """Populate ``dist`` of each :class:`Box` using the median depth inside.

        ``depth_map`` is expected to be the output of
        :meth:`relative_to_absolute_depth`.
        """

        if depth_map.size == 0:
            return boxes

        h, w = depth_map.shape
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(0, min(w - 1, int(x2)))
            y2 = max(0, min(h - 1, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = depth_map[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            box.dist = float(np.nanmedian(crop))
        return boxes

    def _write_depth_artifacts(self, image_path: str, depth_map: np.ndarray) -> tuple[Optional[str], Optional[str]]:
        if depth_map.size == 0:
            return None, None
        normalized = self._normalize_depth_map(depth_map)
        stem = Path(image_path).stem
        timestamp = int(time.time() * 1000)
        suffix = f"{stem}_{timestamp}"
        depth_path = self._depth_dir / f"{suffix}_depth.png"
        cv2.imwrite(str(depth_path), normalized)
        overlay = self._build_overlay(image_path, normalized)
        overlay_path: Optional[str] = None
        if overlay is not None:
            overlay_dest = self._depth_dir / f"{suffix}_overlay.png"
            cv2.imwrite(str(overlay_dest), overlay)
            overlay_path = str(overlay_dest)
        return str(depth_path), overlay_path

    @staticmethod
    def _normalize_depth_map(depth_map: np.ndarray) -> np.ndarray:
        if depth_map.size == 0:
            return np.zeros_like(depth_map, dtype="uint8")
        finite = depth_map[np.isfinite(depth_map)]
        if finite.size == 0:
            return np.zeros_like(depth_map, dtype="uint8")
        d_min = float(np.nanmin(finite))
        d_max = float(np.nanmax(finite))
        if d_max <= d_min:
            return np.zeros_like(depth_map, dtype="uint8")
        safe_map = np.nan_to_num(depth_map, nan=d_min, neginf=d_min, posinf=d_max)
        span = d_max - d_min
        scaled = np.clip((safe_map - d_min) / span, 0.0, 1.0)
        normalized = (scaled * 255.0).astype("uint8")
        return normalized

    def _build_overlay(self, image_path: str, depth_norm: np.ndarray) -> Optional[np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            return None
        height, width = image.shape[:2]
        if depth_norm.shape != (height, width):
            depth_norm = cv2.resize(depth_norm, (width, height), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0.0)
        return overlay

    @property
    def depth_error(self) -> Optional[str]:
        return self._depth_error

