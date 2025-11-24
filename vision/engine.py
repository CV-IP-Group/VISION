
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
        focus = kwargs.pop("focus", None)
        detections = self._backends.detect_objects(image_path, **kwargs)

        depth_map: Optional[np.ndarray] = None
        depth_raw_path: Optional[str] = None
        depth_overlay_path: Optional[str] = None
        if self._backend_depth_error() is None:
            lidar_dist = self._extract_forward_distance(lidar_snapshot)
            depth_map = self._resolve_depth_map(image_path, rel_depth, abs_depth, lidar_dist)
            if depth_map is not None and depth_map.size:
                RealBackends.attach_depth_to_boxes(detections, depth_map)
                if hasattr(self._backends, "_write_depth_artifacts"):
                    try:
                        depth_raw_path, depth_overlay_path = self._backends._write_depth_artifacts(
                            image_path,
                            depth_map,
                        )
                    except Exception:
                        depth_raw_path, depth_overlay_path = None, None

        if focus in {"left", "right", "center"} and detections:
            img_width = self._read_image_width(image_path)
            filtered: list[Box] = []
            for box in detections:
                x1, _, x2, _ = box.xyxy
                cx = (x1 + x2) / 2
                region = self._classify_horizontal_region(img_width, cx, center_ratio=1.0 / 3.0)
                if region == focus:
                    filtered.append(box)
            detections = filtered

        meta: Dict[str, Any] = {"stub": isinstance(self._backends, StubBackends)}
        self._attach_depth_meta(meta, depth_map)
        if depth_raw_path:
            meta["depth_raw_path"] = depth_raw_path
        if depth_overlay_path:
            meta["depth_overlay_path"] = depth_overlay_path
        yolo_debug_path = self._run_dir / "debug_yolo_input.jpg"
        if yolo_debug_path.exists():
            meta["yolo_debug_path"] = str(yolo_debug_path)

        if depth_map is not None and depth_map.size:
            median_val = float(np.nanmedian(depth_map))
            if np.isfinite(median_val):
                meta["scene_distance_m"] = median_val

        return VisionResult(
            task="vision.detect_objects",
            image_path=image_path,
            objects=detections,
            caption=self._build_object_caption(detections),
            meta=meta,
        )

    def check_hazards(self, image_path: str, **kwargs: Any) -> VisionResult:
        lidar_snapshot = kwargs.pop("lidar_snapshot", None)
        rel_depth = kwargs.pop("rel_depth", None)
        abs_depth = kwargs.pop("abs_depth", None)
        boxes = kwargs.pop("boxes", None)

        hazard_info = self._handle_check_hazards(
            image_path=image_path,
            lidar_snapshot=lidar_snapshot,
            rel_depth=rel_depth,
            abs_depth=abs_depth,
            boxes=boxes,
        )
        caption = hazard_info.get("message", "未检测到危险。")
        meta: Dict[str, Any] = hazard_info.get("meta", {})
        depth_map = abs_depth if abs_depth is not None else rel_depth
        depth_raw_path: Optional[str] = None
        depth_overlay_path: Optional[str] = None
        if depth_map is not None and depth_map.size and hasattr(self._backends, "_write_depth_artifacts"):
            try:
                depth_raw_path, depth_overlay_path = self._backends._write_depth_artifacts(
                    image_path,
                    depth_map,
                )
            except Exception:
                depth_raw_path, depth_overlay_path = None, None
        self._attach_depth_meta(meta, depth_map)
        if depth_raw_path:
            meta["depth_raw_path"] = depth_raw_path
        if depth_overlay_path:
            meta["depth_overlay_path"] = depth_overlay_path
        return VisionResult(
            task="vision.check_hazards",
            image_path=image_path,
            caption=caption,
            message=caption,
            objects=boxes or [],
            meta=meta,
        )

    def _handle_check_hazards(
        self,
        image_path: str,
        lidar_snapshot: Optional[Dict[str, Any]],
        rel_depth: Optional[np.ndarray],
        abs_depth: Optional[np.ndarray],
        boxes: Optional[list[Box]] = None,
    ) -> Dict[str, Any]:
        """基于视觉深度和 YOLO 距离统一评估危险，LiDAR 仅用于深度标定。"""

        hazard_meta: Dict[str, Any] = {"stub": isinstance(self._backends, StubBackends)}
        hazard_level = "safe"
        status = "ok"
        message = "未检测到危险。"

        depth_map = abs_depth
        if depth_map is None and rel_depth is not None:
            depth_map = self._backends.calibrated_absolute_depth(rel_depth, None)

        visual_min_distance: Optional[float] = None
        if depth_map is not None and depth_map.size:
            visual_min_distance = self._front_min_distance_from_depth(depth_map)
            if visual_min_distance is not None:
                hazard_meta["visual_min_distance"] = visual_min_distance

        nearest_box_dist: Optional[float] = None
        nearest_box_cls: Optional[str] = None
        detected_meta: list[Dict[str, Any]] = []

        if boxes:
            hazard_meta["detected_box_count"] = len(boxes)
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy
                entry = {
                    "xyxy": box.xyxy,
                    "cls": box.cls,
                    "score": box.score,
                    "dist": box.dist,
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                }
                detected_meta.append(entry)
                if isinstance(box.dist, (int, float)) and np.isfinite(box.dist):
                    d_val = float(box.dist)
                    if nearest_box_dist is None or d_val < nearest_box_dist:
                        nearest_box_dist = d_val
                        nearest_box_cls = box.cls or "物体"
            hazard_meta["detected_boxes"] = detected_meta

        if nearest_box_dist is not None:
            d = float(nearest_box_dist)
            name = nearest_box_cls or "物体"
            if d < 0.5:
                hazard_level = "danger"
                message = f"⚠ 前方 {d:.2f} 米有一个 {name}，距离非常近，请立即注意。"
            elif d < 1.0:
                hazard_level = "warning"
                message = f"注意，前方 {d:.2f} 米有一个 {name}。"
            else:
                message = f"前方约 {d:.2f} 米有一个 {name}。"
        elif visual_min_distance is not None:
            d = float(visual_min_distance)
            if d < 0.5:
                hazard_level = "danger"
                message = f"⚠ 前方约 {d:.2f} 米存在近距离障碍物。"
            elif d < 1.0:
                hazard_level = "warning"
                message = f"注意，前方约 {d:.2f} 米有物体。"
            else:
                message = f"前方约 {d:.2f} 米有物体。"

        if boxes:
            perception_message = self._build_hazard_perception_message(image_path, boxes)
            semantic_sentence = self._build_hazard_semantic_sentence(
                detected_meta,
                visual_min_distance,
            )
            if perception_message:
                message = f"{message} {perception_message}".strip()
            # If desired, semantic_sentence can be appended here. Keeping primary message concise.

        if hazard_level != "safe":
            status = "warn"

        hazard_meta["hazard_level"] = hazard_level
        hazard_meta["hazard_status"] = status

        return {"status": status, "message": message, "meta": hazard_meta}

    def _build_hazard_semantic_sentence(
        self,
        detected: list[Dict[str, Any]],
        fallback_dist: Optional[float],
    ) -> str:
        if not detected:
            return ""
        candidates = [box for box in detected if isinstance(box.get("dist"), (int, float))]
        if not candidates:
            return ""
        primary = min(candidates, key=lambda box: box["dist"])
        cls_name = primary.get("cls") or "物体"
        dist = primary.get("dist")
        if dist is None:
            dist = fallback_dist
        if dist is None:
            return f"注意，前方有一个 {cls_name}。"
        return f"注意，前方 {float(dist):.2f} 米有一个 {cls_name}。"

    def _build_hazard_perception_message(self, image_path: str, boxes: list[Box]) -> str:
        """Describe the closest central object for hazard awareness."""

        if not boxes:
            return ""
        img_width = self._read_image_width(image_path)
        width = max(img_width, 1)
        third = width / 3.0
        center_start = third
        center_end = width - third
        candidates: list[tuple[float, Box, float]] = []
        for box in boxes:
            if box.dist is None or not box.cls:
                continue
            x1, _, x2, _ = box.xyxy
            cx = (x1 + x2) / 2
            if center_start <= cx <= center_end:
                candidates.append((box.dist, box, cx))
        if not candidates:
            return ""
        candidates.sort(key=lambda entry: entry[0])
        dist, best_box, cx = candidates[0]
        region = self._classify_horizontal_region(width, cx, center_ratio=1.0 / 3.0)
        direction_text = self._format_direction_label(region, fine=True)
        return f"注意，{direction_text} {dist:.2f} 米有一个 {best_box.cls}。"

    def _front_min_distance_from_depth(self, depth_map: Optional[np.ndarray]) -> Optional[float]:
        if depth_map is None or not depth_map.size:
            return None
        _, width = depth_map.shape
        front_width = max(1, int(width * 0.3))
        start = max(0, (width - front_width) // 2)
        region = depth_map[:, start : start + front_width]
        mask = np.isfinite(region) & (region > 0)
        if not mask.any():
            return None
        return float(np.min(region[mask]))

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
        if depth_path:
            meta["depth_raw_path"] = depth_path
        if overlay_path:
            meta["depth_overlay_path"] = overlay_path
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

        for box in objects:
            x1, _, x2, _ = box.xyxy
            cx = (x1 + x2) / 2
            region = self._classify_horizontal_region(img_width, cx, center_ratio=1.0 / 3.0)
            if region == "left":
                left_objs.append(box)
            elif region == "right":
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
        result.task = "vision.describe_scene"
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

        forward_min_distance = self._extract_forward_distance(lidar_snapshot)

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
        yolo_boxes: Optional[list[Box]] = None
        if depth_error is None:
            depth_for_boxes = abs_depth if abs_depth is not None else rel_depth
            if depth_for_boxes is not None and depth_for_boxes.size:
                yolo_boxes = self._backends.detect_objects(image_path, conf=yolo_conf)
                if yolo_boxes:
                    RealBackends.attach_depth_to_boxes(yolo_boxes, depth_for_boxes)

        scene_distance: Optional[float] = None
        visual_forward: Optional[float] = None
        if abs_depth is not None and abs_depth.size:
            try:
                _, width = abs_depth.shape
                start = width // 3
                end = max(start + 1, (2 * width) // 3)
                center_slice = abs_depth[:, start:end]
                finite = center_slice[np.isfinite(center_slice) & (center_slice > 0)]
                if finite.size > 20:
                    visual_forward = float(np.percentile(finite, 10))
            except ValueError:
                pass

        if visual_forward is not None:
            scene_distance = visual_forward

        if forward_min_distance is not None and np.isfinite(forward_min_distance):
            scene_distance = (
                forward_min_distance
                if scene_distance is None
                else min(scene_distance, forward_min_distance)
            )

        result = self.check_hazards(
            image_path,
            lidar_snapshot=lidar_snapshot,
            rel_depth=rel_depth,
            abs_depth=abs_depth,
            boxes=yolo_boxes,
            **backend_kwargs,
        )

        fusion_meta: Dict[str, Any] = {
            "applied_thresholds": {
                "yolo_conf_threshold": yolo_conf,
            },
        }
        if forward_min_distance is not None:
            fusion_meta["forward_min_distance"] = forward_min_distance
        if scene_distance is not None:
            fusion_meta["scene_distance_m"] = scene_distance
        if visual_forward is not None:
            fusion_meta["visual_forward_distance"] = visual_forward

        result.meta.update({k: v for k, v in fusion_meta.items() if v is not None})
        if lidar_snapshot is not None:
            result.meta.setdefault("lidar_snapshot", lidar_snapshot)

        self._attach_depth_meta(result.meta, abs_depth)
        final_hazard_level = result.meta.get("hazard_level")
        if final_hazard_level is not None:
            fusion_meta["hazard_level"] = final_hazard_level
        final_status = "warn" if final_hazard_level in {"danger", "warning"} else "ok"
        return {
            "status": final_status,
            "message": result.message,
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
            return "当前画面未检测到目标。"
        labels = {box.cls for box in detections if box.cls}
        summary = join_strings(labels)
        return f"画面中可能包含：{summary}" if summary else None

    @staticmethod
    def _read_image_width(image_path: str) -> int:
        try:
            with Image.open(image_path) as img:
                width = img.width
        except Exception:
            return 1
        return max(1, width)

    def _classify_horizontal_region(
        self,
        img_width: int,
        cx: float,
        *,
        center_ratio: float = 1.0 / 3.0,
    ) -> str:
        width = max(img_width, 1)
        center_width = width * center_ratio
        side_width = (width - center_width) / 2.0
        left_end = side_width
        right_start = width - side_width
        if cx < left_end:
            return "left"
        if cx > right_start:
            return "right"
        return "center"

    def _format_direction_label(self, region: str, *, fine: bool = False) -> str:
        if not fine:
            if region == "left":
                return "左侧"
            if region == "right":
                return "右侧"
            return "前方"
        else:
            if region == "left":
                return "略偏左侧"
            if region == "right":
                return "略偏右侧"
            return "正前方"

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
