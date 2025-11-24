
"""Task orchestrator for VisionMate (P0 API)."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
import json
import math

import frame_capture
from lidar.lidar_bridge import read_latest_snapshot
from scripts.infer_sklearn_intent import (
    MAP as INTENT_MAP,
    keyword_route,
    load_model,
    route_mcp,
)
from vision.engine import CVModuleProcess
from vision.types import Box, VisionResult
from vision.utils import optional_camera_index
import traceback

ResolutionKey = Literal["low", "medium", "high"]
CaptureAttempt = Dict[str, Any]
DEFAULT_CAMERA_INDEX_ENV = "VISIONMATE_CAMERA_INDEX"


class Orchestrator:
    """High-level coordinator linking text intents to CV tasks."""

    RESOLUTION_STRATEGY: Dict[str, ResolutionKey] = {
        "vision.ocr_read": "high",
        "vision.detect_objects": "medium",
        "vision.check_hazards": "low",
        "vision.distance_query": "medium",
        "vision.describe_scene": "medium",
    }

    _capture_sequences: Dict[ResolutionKey, Tuple[Tuple[str, Tuple[int, int]], ...]] = {
        "high": (
            ("high", (1920, 1080)),
            ("medium", (1280, 720)),
            ("low", (640, 480)),
        ),
        "medium": (
            ("medium", (1280, 720)),
            ("low", (640, 480)),
        ),
        "low": (
            ("low", (640, 480)),
        ),
    }

    ENTITY_MAP: Dict[str, str] = {
        # Furniture
        "椅子": "chair",
        "凳": "chair",
        "桌": "dining table",
        "桌子": "dining table",
        "沙发": "couch",
        "床": "bed",
        # Electronics
        "电脑": "laptop",
        "本": "laptop",
        "PC": "laptop",
        "手机": "cell phone",
        "电话": "cell phone",
        "手提": "cell phone",
        "电视": "tv",
        "屏幕": "tv",
        "显示器": "tv",
        "鼠标": "mouse",
        "键盘": "keyboard",
        # Personal
        "人": "person",
        "家伙": "person",
        "男": "person",
        "女": "person",
        "手提包": "handbag",
        "手袋": "handbag",
        "箱": "suitcase",
        "伞": "umbrella",
        # Kitchen
        "杯": "cup",
        "杯子": "cup",
        "水杯": "cup",
        "瓶": "bottle",
        "碗": "bowl",
        # Vehicles
        "汽车": "car",
        "巴士": "bus",
        "公交": "bus",
        "单车": "bicycle",
    }

    _LIDAR_UNAVAILABLE_NOTE = "当前无 LiDAR 快照，估算可能存在误差。"

    def __init__(
        self,
        model_path: str = "models/intent_sklearn.joblib",
        threshold_path: str = "models/intent_threshold.json",
        default_camera_index: Optional[int] = None,
        capture_dir: str | None = None,
        run_dir: str | None = None,
        weights_dir: str | None = None,
    ) -> None:
        self._intent_model_path = Path(model_path)
        self._intent_threshold_path = Path(threshold_path)
        self._model, self._tau = self._load_intent_model()
        self._default_camera_index = self._resolve_default_camera_index(default_camera_index)
        self._capture_dir = capture_dir or "captured_images"
        self._run_dir = run_dir or "runs"
        self._weights_dir = weights_dir or "weights"
        self._cv = CVModuleProcess(
            weights_dir=self._weights_dir,
            run_dir=self._run_dir,
        )

    def handle_text(self, text: str, camera_index: Optional[int] = None) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"timings": {}}
        start_total = time.perf_counter()

        # lidar_snapshot: Optional[Dict[str, Any]] = None
        # --- NEW VERSION ---
        lidar_context = self._probe_lidar_snapshot()
        lidar_snapshot = lidar_context.get("data")

        if not text:
            # meta["timings"]["total"] = time.perf_counter() - start_total
            # --- NEW VERSION ---
            total_elapsed = time.perf_counter() - start_total
            self._record_timing(meta["timings"], "total", total_elapsed)
            self._attach_lidar_snapshot(meta, lidar_context)
            return {
                "status": "warn",
                "message": "没有收到有效的指令。",
                "result": None,
                "intent": None,
                "meta": meta,
            }

        intent_start = time.perf_counter()

        # 关键词优先路由（规则优先，模型兜底）——避免对课堂演示依赖外部训练模型
        text_norm = (text or "").strip().lower()
        intent_spec = keyword_route(text_norm)
        try:
            if intent_spec is None:
                if getattr(self, "_model", None) is None:
                    intent_spec = {
                        "action": "warn",
                        "msg": "意图分类模型不可用，请使用关键词（如“读”“距”“险”“前”）进行查询。",
                    }
                else:
                    intent_spec = route_mcp(
                        text,
                        self._model,
                        self._tau,
                        INTENT_MAP,
                        skip_rules=True,
                    )
        except Exception as exc:  # noqa: BLE001
            intent_elapsed = time.perf_counter() - intent_start
            self._record_timing(meta["timings"], "intent", intent_elapsed)
            total_elapsed = time.perf_counter() - start_total
            self._record_timing(meta["timings"], "total", total_elapsed)
            meta["error"] = f"intent-routing: {exc}"
            self._attach_lidar_snapshot(meta, lidar_context)
            return {
                "status": "error",
                "message": "任务分类失败，请稍后再试。",
                "result": None,
                "intent": None,
                "meta": meta,
            }

        # 记录 intent 耗时（规则匹配也计时）
        intent_elapsed = time.perf_counter() - intent_start
        self._record_timing(meta["timings"], "intent", intent_elapsed)

        if intent_spec is None:
            print("[WARN] No intent match. Defaulting to 'clarify'.")
            intent_spec = {
                "action": "clarify",
                "message": "抱歉，我没听懂，请再说一遍。",
                "confidence": 0.0,
            }

        if intent_spec.get("action") != "call":
            # meta["timings"]["total"] = time.perf_counter() - start_total
            # --- NEW VERSION ---
            preview_path, used_resolution, capture_meta = self._safe_capture("medium", camera_index)
            self._record_timing(meta["timings"], "capture", capture_meta["elapsed"])
            self._update_capture_meta(meta, capture_meta, used_resolution)
            if preview_path:
                meta.setdefault("preview_image_path", preview_path)

            total_elapsed = time.perf_counter() - start_total
            self._record_timing(meta["timings"], "total", total_elapsed)
            self._attach_lidar_snapshot(meta, lidar_context)
            return {
                "status": "warn",
                "message": intent_spec.get("msg", "这个请求暂时无法处理。"),
                "result": None,
                "intent": intent_spec,
                "meta": meta,
            }

        function_name = intent_spec.get("function")
        if not function_name:
            # meta["timings"]["total"] = time.perf_counter() - start_total
            # --- NEW VERSION ---
            total_elapsed = time.perf_counter() - start_total
            self._record_timing(meta["timings"], "total", total_elapsed)
            self._attach_lidar_snapshot(meta, lidar_context)
            return {
                "status": "warn",
                "message": "收到不完整的任务说明。",
                "result": None,
                "intent": intent_spec,
                "meta": meta,
            }

        desired_resolution = self.RESOLUTION_STRATEGY.get(function_name, "medium")
        image_path, used_resolution, capture_meta = self._safe_capture(
            desired_resolution,
            camera_index,
        )
        # meta["timings"]["capture"] = capture_meta["elapsed"]
        # --- NEW VERSION ---
        capture_elapsed = capture_meta["elapsed"]
        self._record_timing(meta["timings"], "capture", capture_elapsed)
        meta["capture_attempts"] = capture_meta["attempts"]
        if used_resolution is not None:
            meta["chosen_resolution"] = used_resolution

        if image_path is None:
            # meta["timings"]["total"] = time.perf_counter() - start_total
            # --- NEW VERSION ---
            total_elapsed = time.perf_counter() - start_total
            self._record_timing(meta["timings"], "total", total_elapsed)
            return {
                "status": "warn",
                "message": "摄像头暂时不可用，请检查设备后重试。",
                "result": None,
                "intent": intent_spec,
                "meta": meta,
            }

        args = dict(intent_spec.get("args", {}))
        slot_kwargs: Dict[str, str] = {}
        # Support both Chinese and English direction hints.
        if re.search(r"(左|left)", text_norm):
            slot_kwargs["focus"] = "left"
        elif re.search(r"(右|right)", text_norm):
            slot_kwargs["focus"] = "right"
        # Dynamic Entity Extraction
        # Iterate through the map to find any matching keyword in the user's text
        for keyword, coco_label in self.ENTITY_MAP.items():
            if keyword.lower() in text_norm:
                slot_kwargs["object"] = coco_label
                # Optional: break after first match to avoid ambiguity, or keep last match
                break
        if slot_kwargs:
            args.update(slot_kwargs)
        cv_start = time.perf_counter()
        try:
            cv_payload = self._cv.run_task(
                function_name,
                image_path,
                lidar_snapshot,
                **args,
            )
        except Exception as exc:  # noqa: BLE001
            # meta["timings"]["cv"] = time.perf_counter() - cv_start
            # meta["timings"]["total"] = time.perf_counter() - start_total
            # --- NEW VERSION ---
            cv_elapsed = time.perf_counter() - cv_start
            self._record_timing(meta["timings"], "cv", cv_elapsed)
            total_elapsed = time.perf_counter() - start_total
            self._record_timing(meta["timings"], "total", total_elapsed)
            meta["error"] = f"cv-execution: {exc}"
            self._attach_lidar_snapshot(meta, lidar_context)
            print("[Orchestrator] cv-execution", exc)
            traceback.print_exc()
            return {
                "status": "error",
                "message": "视觉模块执行失败，请稍后再试。",
                "result": None,
                "intent": intent_spec,
                "meta": meta,
            }
        # meta["timings"]["cv"] = time.perf_counter() - cv_start
        # --- NEW VERSION ---
        cv_elapsed = time.perf_counter() - cv_start
        self._record_timing(meta["timings"], "cv", cv_elapsed)

        cv_status = cv_payload.get("status", "ok") if isinstance(cv_payload, dict) else "ok"
        result = cv_payload.get("result") if isinstance(cv_payload, dict) else None
        fusion_meta = cv_payload.get("meta", {}) if isinstance(cv_payload, dict) else {}
        message_override = cv_payload.get("message") if isinstance(cv_payload, dict) else None

        if fusion_meta:
            meta.update(fusion_meta)

        if cv_status == "error":
            total_elapsed = time.perf_counter() - start_total
            self._record_timing(meta["timings"], "total", total_elapsed)
            self._attach_lidar_snapshot(meta, lidar_context)
            return {
                "status": "error",
                "message": message_override or "视觉模块执行失败，请稍后再试。",
                "result": None,
                "intent": intent_spec,
                "meta": meta,
            }

        if result is None:
            total_elapsed = time.perf_counter() - start_total
            self._record_timing(meta["timings"], "total", total_elapsed)
            self._attach_lidar_snapshot(meta, lidar_context)
            return {
                "status": "warn",
                "message": message_override or "视觉结果缺失，请稍后再试。",
                "result": None,
                "intent": intent_spec,
                "meta": meta,
            }

        target_object = args.get("object")
        message = self._compose_message(result, target_object=target_object)
        # 注入耗时信息到视觉结果的 meta，供 data.meta.timings 使用。
        # timings_copy = dict(meta.get("timings", {}))
        # result.meta.setdefault("timings", {}).update(timings_copy)
        # meta["timings"]["total"] = time.perf_counter() - start_total
        # --- NEW VERSION ---
        total_elapsed = time.perf_counter() - start_total
        self._record_timing(meta["timings"], "total", total_elapsed)
        result.meta.setdefault("timings", {}).update(dict(meta.get("timings", {})))
        if function_name:
            result.meta["intent"] = function_name
        if used_resolution is not None:
            result.meta["capture_resolution"] = used_resolution
        lidar_payload = self._attach_lidar_snapshot(meta, lidar_context)
        if lidar_payload is not None:
            result.meta.setdefault("lidar_snapshot", lidar_payload)

        if fusion_meta:
            result.meta.update({k: v for k, v in fusion_meta.items() if k not in {"error_code"}})

        final_message = message_override or message
        image_result_path = getattr(result, "image_path", None)
        vis_meta = result.meta if isinstance(getattr(result, "meta", None), dict) else {}
        status_dump = {
            "timestamp": time.time(),
            "intent": function_name or "unknown",
            "raw_text": text,
            "message": final_message,
            "image_path": image_result_path,
            "hazard_level": meta.get("hazard_level", "safe"),
            "timings": meta.get("timings", {}),
            "yolo_debug_path": vis_meta.get("yolo_debug_path"),
            "depth_overlay_path": vis_meta.get("depth_overlay_path"),
            "depth_raw_path": vis_meta.get("depth_raw_path"),
        }
        status_dir = Path(self._run_dir)
        status_dir.mkdir(parents=True, exist_ok=True)
        with open(status_dir / "latest_status.json", "w", encoding="utf-8") as f:
            json.dump(status_dump, f, ensure_ascii=False, indent=2)
        return {
            "status": "ok",
            "message": final_message,
            "result": result,
            "intent": intent_spec,
            "meta": meta,
        }

    def _resolve_handler(self, function_name: str):
        callbacks = {
            "vision.detect_objects": self._cv.detect_objects,
            "vision.check_hazards": self._cv.check_hazards,
            "vision.distance_query": self._cv.distance_query,
            "vision.describe_scene": self._cv.describe_scene,
            "vision.ocr_read": self._cv.ocr_read,
        }
        return callbacks.get(function_name)

    def _safe_capture(
        self,
        desired_resolution: ResolutionKey,
        camera_index: Optional[int],
    ) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        attempts: List[CaptureAttempt] = []
        capture_start = time.perf_counter()
        cam_idx = optional_camera_index(camera_index, self._default_camera_index)
        sequence = self._capture_sequences.get(desired_resolution, self._capture_sequences["low"])

        for label, dims in sequence:
            attempt_start = time.perf_counter()
            try:
                image_path = frame_capture.capture_image(dims, label, cam_idx, self._capture_dir)
            except Exception:  # noqa: BLE001
                image_path = None
            attempt_elapsed = time.perf_counter() - attempt_start
            attempts.append({"label": label, "success": bool(image_path), "elapsed": attempt_elapsed})
            if image_path:
                return image_path, label, {"elapsed": time.perf_counter() - capture_start, "attempts": attempts}

        return None, None, {"elapsed": time.perf_counter() - capture_start, "attempts": attempts}

    def _compose_message(
        self,
        result: VisionResult,
        target_object: Optional[str] = None,
    ) -> str:
        fragments: list[str] = []
        caption_has_distance = False
        base_message_added = False

        if result.task == "vision.ocr_read":
            if result.ocr_text:
                fragments.append(f"OCR 读到：{result.ocr_text}")
            else:
                fragments.append("未在画面中识别到文字。")
            base_message_added = True
        else:
            labels = []
            if result.objects:
                labels = sorted({box.cls for box in result.objects if box.cls})

            if target_object:
                focus_boxes = [box for box in (result.objects or []) if box.cls == target_object]
                if focus_boxes:
                    fragments.append(f"找到了 {target_object}，共 {len(focus_boxes)} 个。")
                else:
                    if labels:
                        others = ", ".join(labels)
                        fragments.append(
                            f"画面里没有找到 {target_object}，但检测到：{others}。"
                        )
                    else:
                        fragments.append(f"画面里没有找到 {target_object}。")
                base_message_added = True

            if not base_message_added:
                if result.caption:
                    fragments.append(result.caption)
                    caption_has_distance = self._caption_contains_distance(result.caption)
                    base_message_added = True
                elif labels:
                    fragments.append(f"检测到：{', '.join(labels)}。")
                    base_message_added = True
                else:
                    fragments.append("当前画面未检测到已知目标。")
                    base_message_added = True

        forward_distance = result.meta.get("forward_min_distance")
        lidar_sentence_added = False
        if isinstance(forward_distance, (int, float)) and math.isfinite(forward_distance):
            if not caption_has_distance:
                dist = float(forward_distance)
                if dist < 0.3:
                    sentence = f"前方 {dist:.2f} 米，危险，极近距离（LiDAR）。"
                elif dist < 0.5:
                    sentence = f"前方 {dist:.2f} 米，非常近（LiDAR）。"
                else:
                    sentence = f"前方 {dist:.2f} 米有障碍物（LiDAR）。"
                fragments.append(sentence)
                lidar_sentence_added = True

        skip_distance = result.task == "vision.ocr_read" or caption_has_distance or lidar_sentence_added
        distance_msg = self._distance_message(result.meta)
        if distance_msg and not skip_distance:
            fragments.append(distance_msg)

        return " ".join(fragments).strip()

    @staticmethod
    def _caption_contains_distance(text: str) -> bool:
        return bool(text and re.search(r"米", text))

    @staticmethod
    def _unique_labels(boxes: Iterable[Box]) -> set[str]:
        return {box.cls for box in boxes if box.cls}

    @staticmethod
    def _record_timing(store: Dict[str, Any], key: str, elapsed: float) -> None:
        """Store both second and millisecond views for timing metrics."""
        store[key] = elapsed
        store[f"{key}_s"] = elapsed
        store[f"{key}_ms"] = int(elapsed * 1000)

    @staticmethod
    def _update_capture_meta(
        meta: Dict[str, Any],
        capture_meta: Dict[str, Any],
        used_resolution: Optional[str],
    ) -> None:
        """Merge capture attempt diagnostics into the response metadata."""
        meta["capture_attempts"] = capture_meta.get("attempts", [])
        if used_resolution is not None:
            meta["chosen_resolution"] = used_resolution

    def _attach_lidar_snapshot(self, meta: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Attach the freshest LiDAR snapshot when available."""
        # snapshot = read_latest_snapshot()
        # if snapshot is not None:
        #     meta["lidar_snapshot"] = snapshot
        # --- NEW VERSION ---
        payload: Dict[str, Any] = {
            "fresh": False,
            "data": None,
            "snapshot_age_ms": None,
        }
        if context:
            payload.update({
                "fresh": bool(context.get("fresh")),
                "data": context.get("data"),
                "snapshot_age_ms": context.get("snapshot_age_ms"),
            })
        snapshot_data = payload.get("data")
        if isinstance(snapshot_data, dict):
            payload.update(snapshot_data)
        if snapshot_data is None:
            payload.pop("data", None)
        meta["lidar_snapshot"] = payload
        return payload

    def _probe_lidar_snapshot(self) -> Dict[str, Any]:
        """Read and cache the LiDAR snapshot together with freshness metadata."""

        snapshot_path = os.getenv("VISIONMATE_LIDAR_SNAPSHOT", "runs/lidar_snapshot.json")
        ttl_raw = os.getenv("VISIONMATE_LIDAR_TTL", "0.5")
        try:
            ttl_seconds = float(ttl_raw)
        except ValueError:
            ttl_seconds = 0.5

        snapshot = read_latest_snapshot(snapshot_path, ttl_seconds)
        age_ms: Optional[int]
        try:
            stat = Path(snapshot_path).stat()
            age_ms = int((time.time() - stat.st_mtime) * 1000)
        except (FileNotFoundError, OSError):
            age_ms = None

        return {
            "data": snapshot,
            "fresh": snapshot is not None,
            "snapshot_age_ms": age_ms,
            "path": snapshot_path,
            "ttl_seconds": ttl_seconds,
        }

    @staticmethod
    def _resolve_default_camera_index(default_camera_index: Optional[int]) -> int:
        env_value = os.getenv(DEFAULT_CAMERA_INDEX_ENV)
        if env_value is not None:
            try:
                return int(env_value)
            except ValueError:
                pass
        if default_camera_index is not None:
            return default_camera_index
        return 0

    def _load_intent_model(self) -> tuple[Optional[Any], Dict[str, Any]]:
        if not self._intent_model_path.exists() or not self._intent_threshold_path.exists():
            self._train_intent_model()
        try:
            return load_model(str(self._intent_model_path), str(self._intent_threshold_path))
        except Exception as exc:
            print("[Orchestrator] intent model load failed, retraining", exc)
            self._train_intent_model()
            try:
                return load_model(str(self._intent_model_path), str(self._intent_threshold_path))
            except Exception as exc2:  # noqa: BLE001
                print("[Orchestrator] intent model unavailable", exc2)
                return None, {}

    def _train_intent_model(self) -> None:
        try:
            from scripts.train_sklearn_intent import main as train_intent_model
        except Exception as exc:  # noqa: BLE001
            print("[Orchestrator] intent trainer import failed", exc)
            return
        try:
            train_intent_model()
        except Exception as exc:  # noqa: BLE001
            print("[Orchestrator] intent trainer execution failed", exc)

    @staticmethod
    def _distance_message(meta: Dict[str, Any]) -> Optional[str]:
        if not isinstance(meta, dict):
            return None

        lidar_distance = Orchestrator._extract_forward_distance_from_meta(meta)
        midas_distance = meta.get("distance_m")
        if midas_distance is None:
            midas_distance = meta.get("scene_distance_m")

        if lidar_distance is None and midas_distance is None:
            return None

        if lidar_distance is not None:
            return f"前方最近 {lidar_distance:.1f} 米（LiDAR 读数）。"

        fragments = [f"前方约 {midas_distance:.1f} 米（MiDaS 估算）。"]
        if not Orchestrator._has_lidar_snapshot(meta):
            fragments.append(Orchestrator._LIDAR_UNAVAILABLE_NOTE)
        return " ".join(fragments)

    @staticmethod
    def _has_lidar_snapshot(meta: Dict[str, Any]) -> bool:
        if meta.get("forward_min_distance") is not None:
            return True
        lidar_snapshot = meta.get("lidar_snapshot")
        if isinstance(lidar_snapshot, dict):
            if lidar_snapshot.get("data") is not None:
                return True
        return False

    @staticmethod
    def _extract_forward_distance_from_meta(meta: Dict[str, Any]) -> Optional[float]:
        candidate = meta.get("forward_min_distance")
        if candidate is None:
            lidar_snapshot = meta.get("lidar_snapshot")
            if isinstance(lidar_snapshot, dict):
                candidate = lidar_snapshot.get("forward_min_distance")
                if candidate is None:
                    data = lidar_snapshot.get("data")
                    if isinstance(data, dict):
                        candidate = data.get("forward_min_distance")
        try:
            return float(candidate) if candidate is not None else None
        except (TypeError, ValueError):
            return None
