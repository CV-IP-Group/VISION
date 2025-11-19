"""Central CV module process with pluggable task handlers.

The default implementation now uses :class:`RealBackends` when available,
falling back to :class:`StubBackends` in tests or when weights are missing.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import logging
import numpy as np
from PIL import Image

from config import SIMULATION_MODE
from .backends import RealBackends, StubBackends
from .types import Box, VisionResult
from .utils import join_strings


class CVModuleProcess:
    """Facade that exposes computer vision task entry points."""

    def __init__(
        self,
        backends: Optional[StubBackends] = None,
        *,
        weights_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        """Initialise the CV engine.

        When ``backends`` is not provided, the engine tries to construct a
        :class:`RealBackends` instance. If the required weights are missing, it
        gracefully falls back to :class:`StubBackends`, keeping the rest of the
        stack operational.
        """

        self._run_dir = Path(run_dir) if run_dir is not None else Path("runs")
        if backends is not None:
            self._backends = backends
            return

        try:
            self._backends = RealBackends(
                weights_dir=weights_dir,
                output_dir=self._run_dir,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            if not SIMULATION_MODE:
                logging.exception("Failed to initialise RealBackends in real mode")
                raise
            logging.warning("RealBackends unavailable, falling back to StubBackends: %s", exc)
            self._backends = StubBackends()

    def detect_objects(self, image_path: str, **kwargs: Any) -> VisionResult:
        rel_depth = kwargs.pop("rel_depth", None)
        abs_depth = kwargs.pop("abs_depth", None)
        lidar_snapshot = kwargs.pop("lidar_snapshot", None)
        detections = self._backends.detect_objects(image_path, **kwargs)

        depth_map: Optional[np.ndarray] = None
        if self._backend_depth_error() is None:
            lidar_dist = self._extract_forward_distance(lidar_snapshot)
            depth_map = self._resolve_depth_map(image_path, rel_depth, abs_depth, lidar_dist)
            if depth_map is not None and depth_map.size:
                RealBackends.attach_depth_to_boxes(detections, depth_map)

        meta: Dict[str, Any] = {"stub": isinstance(self._backends, StubBackends)}
        self._attach_depth_meta(meta, depth_map)

        return VisionResult(
            task="vision.detect_objects",
            image_path=image_path,
            objects=detections,
            caption=self._build_object_caption(detections),
            meta=meta,
        )

    def check_hazards(self, image_path: str, **kwargs: Any) -> VisionResult:
        hazard_info = self._backends.check_hazards(image_path, **kwargs)
        caption = hazard_info.get("message", "")
        meta: Dict[str, Any] = {
            "stub": isinstance(self._backends, StubBackends),
            "status": hazard_info.get("status"),
        }
        self._attach_depth_meta(meta, None)
        return VisionResult(
            task="vision.check_hazards",
            image_path=image_path,
            caption=caption or "危险检测结果。",
            meta=meta,
        )

    def distance_query(self, image_path: str, **kwargs: Any) -> VisionResult:
        distance_info = self._backends.distance_query(image_path, **kwargs)
        caption = distance_info.get("message", "")
        depth_path = distance_info.get("depth_path")
        overlay_path = distance_info.get("overlay_path")
        meta: Dict[str, Any] = {"stub": isinstance(self._backends, StubBackends)}
        if distance_info.get("distance_m") is not None:
            meta["distance_m"] = distance_info["distance_m"]
        if depth_path or overlay_path:
            meta["depth_available"] = True
        result = VisionResult(
            task="vision.distance_query",
            image_path=image_path,
            caption=caption or "深度估计模型尚未连接。",
            meta=meta,
        )
        if depth_path:
            result.depth_path = depth_path
        if overlay_path:
            result.overlay_path = overlay_path
        self._attach_depth_meta(meta, None)
        return result

    def describe_scene(self, image_path: str, **kwargs: Any) -> VisionResult:
        result = self.detect_objects(image_path, **kwargs)
        objects = result.objects
        img_width = self._read_image_width(image_path)

        left_objs: list[Box] = []
        center_objs: list[Box] = []
        right_objs: list[Box] = []

        third = img_width / 3.0 if img_width > 0 else 1.0
        for box in objects:
            x1, _, x2, _ = box.xyxy
            cx = (x1 + x2) / 2
            if cx < third:
                left_objs.append(box)
            elif cx > img_width - third:
                right_objs.append(box)
            else:
                center_objs.append(box)

        message_fragments: list[str] = []
        center_summary = self._summarize_counts(center_objs)
        if center_summary:
            message_fragments.append(f"正前方发现{center_summary}。")
        left_summary = self._summarize_counts(left_objs)
        if left_summary:
            message_fragments.append(f"左侧有{left_summary}。")
        right_summary = self._summarize_counts(right_objs)
        if right_summary:
            message_fragments.append(f"右侧有{right_summary}。")

        message = (
            "前方视野清晰，未检测到障碍。"
            if not message_fragments
            else " ".join(message_fragments)
        )

        result.caption = message
        result.message = message
        result.meta.setdefault("stub", isinstance(self._backends, StubBackends))
        return result

    def ocr_read(self, image_path: str, **kwargs: Any) -> VisionResult:
        text = self._backends.ocr_read(image_path, **kwargs)
        caption = f"读到的文字是：{text}" if text else "未在画面中识别到文字。"
        return VisionResult(
            task="vision.ocr_read",
            image_path=image_path,
            ocr_text=text,
            caption=caption,
            meta={"stub": isinstance(self._backends, StubBackends)},
        )

    def run_task(
        self,
        function: str,
        image_path: str,
        lidar_snapshot: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Dispatch to the requested CV routine while preserving LiDAR context."""

        callbacks = {
            "vision.detect_objects": self.detect_objects,
            "vision.distance_query": self.distance_query,
            "vision.describe_scene": self.describe_scene,
            "vision.ocr_read": self.ocr_read,
        }

        if function == "vision.check_hazards":
            return self._run_check_hazards(image_path, lidar_snapshot, **kwargs)

        handler = callbacks.get(function)
        # if handler is None:
        #     raise ValueError(f"Unsupported vision task: {function}")
        # --- NEW VERSION ---
        if handler is None:
            return {
                "status": "warn",
                "result": None,
                "meta": {"error_code": "unsupported_function", "function": function},
                "message": "该任务尚未实现，请稍后再试。",
            }

        handler_kwargs = dict(kwargs)
        if lidar_snapshot is not None:
            handler_kwargs["lidar_snapshot"] = lidar_snapshot
        result = handler(image_path=image_path, **handler_kwargs)
        if lidar_snapshot is not None:
            result.meta.setdefault("lidar_snapshot", lidar_snapshot)
        return {
            "status": "ok",
            "result": result,
            "meta": {},
        }

    def _run_check_hazards(
        self,
        image_path: str,
        lidar_snapshot: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Apply Level-0 fusion rules that blend LiDAR priors with vision heuristics."""

        backend_kwargs = dict(kwargs)
        yolo_conf = float(backend_kwargs.pop("yolo_conf_threshold", 0.5))
        midas_threshold = float(backend_kwargs.pop("midas_near_threshold", 1.0))

        forward_min_distance = self._extract_forward_distance(lidar_snapshot)
        hazard_level: Optional[str] = None
        if forward_min_distance is not None:
            if forward_min_distance <= 0.5:
                hazard_level = "high"
            elif forward_min_distance <= 1.5:
                hazard_level = "medium"
                yolo_conf = min(yolo_conf, 0.3)
                midas_threshold = min(midas_threshold, 0.9)

        backend_kwargs.update(
            {
                "yolo_conf_threshold": yolo_conf,
                "midas_near_threshold": midas_threshold,
            }
        )

        depth_error = self._backend_depth_error()
        rel_depth: Optional[np.ndarray] = None
        abs_depth: Optional[np.ndarray] = None
        if depth_error is None:
            rel_depth = self._extract_depth_array(self._backends.estimate_depth(image_path))
            if rel_depth is not None:
                abs_depth = self._backends.calibrated_absolute_depth(
                    rel_depth,
                    forward_min_distance,
                )

        detection_result = self.detect_objects(
            image_path,
            rel_depth=rel_depth,
            abs_depth=abs_depth,
            **backend_kwargs,
        )

        scene_distance: Optional[float] = None
        if abs_depth is not None and abs_depth.size:
            median_value = float(np.nanmedian(abs_depth))
            if not np.isnan(median_value):
                scene_distance = median_value

        result = self.check_hazards(image_path, **backend_kwargs)

        fusion_meta: Dict[str, Any] = {
            "forward_min_distance": forward_min_distance,
            "applied_thresholds": {
                "yolo_conf_threshold": yolo_conf,
                "midas_near_threshold": midas_threshold,
            },
        }
        if scene_distance is not None:
            fusion_meta["scene_distance_m"] = scene_distance
        if hazard_level is not None:
            fusion_meta["hazard_level"] = hazard_level
        if forward_min_distance is None:
            fusion_meta.pop("forward_min_distance")

        result.meta.update({k: v for k, v in fusion_meta.items() if v is not None})
        if forward_min_distance is not None:
            result.meta["forward_min_distance"] = forward_min_distance
        if hazard_level is not None:
            result.meta["hazard_level"] = hazard_level
        if lidar_snapshot is not None:
            result.meta.setdefault("lidar_snapshot", lidar_snapshot)

        self._attach_depth_meta(result.meta, abs_depth)

        message = self._compose_hazard_message(
            detection_result.objects,
            hazard_level,
            forward_min_distance,
            scene_distance,
        )
        result.caption = message
        result.message = message

        return {
            "status": "ok",
            "result": result,
            "meta": {k: v for k, v in fusion_meta.items() if v is not None},
        }

    @staticmethod
    def _extract_forward_distance(lidar_snapshot: Optional[Dict[str, Any]]) -> Optional[float]:
        """Pull the forward minimum distance from raw LiDAR payloads when available."""

        if not isinstance(lidar_snapshot, dict):
            return None

        candidate = lidar_snapshot.get("forward_min_distance")
        if candidate is None and isinstance(lidar_snapshot.get("data"), dict):
            candidate = lidar_snapshot["data"].get("forward_min_distance")

        try:
            return float(candidate) if candidate is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_object_caption(detections: list[Box]) -> Optional[str]:
        if not detections:
            return "未检测到目标（占位结果）。"
        labels = {box.cls for box in detections if box.cls}
        summary = join_strings(labels)
        return f"画面中可能包含：{summary}" if summary else None

    def _compose_hazard_message(
        self,
        boxes: list[Box],
        hazard_level: Optional[str],
        lidar_distance: Optional[float],
        scene_distance: Optional[float],
    ) -> str:
        fragments: list[str] = []
        if hazard_level == "high":
            fragments.append("前方危险等级：高，立刻停止。")
        elif hazard_level == "medium":
            fragments.append("前方危险等级：中等，请小心前行。")
        elif hazard_level == "low":
            fragments.append("前方危险等级：低，继续观察。")

        if lidar_distance is not None:
            fragments.append(f"LiDAR 显示前方最近 {lidar_distance:.1f} 米。")
        elif scene_distance is not None:
            fragments.append(f"MiDaS 估算前方约 {scene_distance:.1f} 米。")
            fragments.append("当前无 LiDAR 快照，估算可能存在误差。")
        else:
            fragments.append("当前无 LiDAR 快照，估算可能存在误差。")

        object_descriptions: list[str] = []
        for box in boxes[:3]:
            label = box.cls or "目标"
            if box.dist is not None:
                object_descriptions.append(f"{label}({box.dist:.1f} 米)")
            else:
                object_descriptions.append(label)
        if object_descriptions:
            fragments.append("视觉检测到" + "、".join(object_descriptions) + "。")

        if fragments:
            return "".join(fragments)
        return "险情检测结果已更新，继续观察前方。"

    @staticmethod
    def _read_image_width(image_path: str) -> int:
        try:
            with Image.open(image_path) as img:
                width = img.width
        except Exception:
            return 1
        return max(1, width)

    @staticmethod
    def _summarize_counts(obj_list: list[Box]) -> str:
        if not obj_list:
            return ""
        counter: Counter[str] = Counter(box.cls for box in obj_list if box.cls)
        if not counter:
            return ""
        parts = [f"{count}个{label}" for label, count in sorted(counter.items())]
        return ", ".join(parts)

    def _attach_depth_meta(self, meta: Dict[str, Any], depth_map: Optional[np.ndarray]) -> None:
        depth_error = self._backend_depth_error()
        if depth_error:
            meta["depth_error"] = depth_error
            return
        if depth_map is not None and depth_map.size:
            meta["depth_available"] = True

    def _backend_depth_error(self) -> Optional[str]:
        return getattr(self._backends, "depth_error", None)

    def _extract_depth_array(self, payload: Any) -> Optional[np.ndarray]:
        if payload is None:
            return None
        candidate = payload[0] if isinstance(payload, tuple) and payload else payload
        return candidate if isinstance(candidate, np.ndarray) else None

    def _resolve_depth_map(
        self,
        image_path: str,
        rel_depth: Optional[np.ndarray],
        abs_depth: Optional[np.ndarray],
        lidar_dist: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        if abs_depth is not None:
            return abs_depth
        if rel_depth is not None:
            return self._backends.calibrated_absolute_depth(rel_depth, lidar_dist)
        extracted = self._extract_depth_array(self._backends.estimate_depth(image_path))
        if extracted is None:
            return None
        return self._backends.calibrated_absolute_depth(extracted, lidar_dist)
