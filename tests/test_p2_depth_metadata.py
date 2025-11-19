"""深度元数据与 Hazard / Detect 检查的验收测试。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vision.engine import CVModuleProcess
from vision.types import Box
from vision.backends import RealBackends, StubBackends


class _DepthAwareBackend(StubBackends):
    def __init__(self) -> None:
        super().__init__()
        self._depth = np.linspace(1.0, 2.0, num=9, dtype="float32").reshape(3, 3)

    def detect_objects(self, image_path: str, **_: Any) -> list[Box]:
        return [
            Box(xyxy=(0, 0, 3, 3), cls="crate", score=0.9),
            Box(xyxy=(0, 0, 1, 1), cls="label", score=0.4),
        ]

    def estimate_depth(self, image_path: str, **_: Any) -> np.ndarray:
        return self._depth


class _BrokenDepthBackend(_DepthAwareBackend):
    @property
    def depth_error(self) -> str:
        return "midas_missing"


def test_detect_objects_applies_depth_and_meta() -> None:
    backend = _DepthAwareBackend()
    engine = CVModuleProcess(backends=backend)

    result = engine.detect_objects("frame.jpg")

    assert result.meta.get("depth_available") is True
    assert result.meta.get("stub") is True
    assert all(box.dist is not None for box in result.objects)
    abs_depth = RealBackends.relative_to_absolute_depth(backend._depth)
    x1, y1, x2, y2 = 0, 0, 3, 3
    w = abs_depth.shape[1]
    h = abs_depth.shape[0]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    crop = abs_depth[y1:y2, x1:x2]
    expected = float(np.nanmedian(crop))
    assert result.objects[0].dist == pytest.approx(expected, rel=1e-6)


def test_detect_objects_reports_depth_error_without_changes() -> None:
    backend = _BrokenDepthBackend()
    engine = CVModuleProcess(backends=backend)

    result = engine.detect_objects("frame.jpg")

    assert result.meta.get("depth_error") == "midas_missing"
    assert "depth_available" not in result.meta
    assert all(box.dist is None for box in result.objects)


def test_hazard_run_task_includes_depth_available() -> None:
    backend = _DepthAwareBackend()
    engine = CVModuleProcess(backends=backend)

    payload = engine.run_task("vision.check_hazards", "frame.jpg", None)

    assert payload["status"] == "ok"
    result = payload["result"]
    assert result is not None
    assert result.meta.get("depth_available") is True
    assert result.meta.get("depth_error") is None
    assert payload["result"].meta.get("scene_distance_m") == pytest.approx(
        float(np.nanmedian(RealBackends.relative_to_absolute_depth(backend._depth))),
        rel=1e-6,
    )
    assert payload["meta"].get("applied_thresholds", {}).get("yolo_conf_threshold") is not None
