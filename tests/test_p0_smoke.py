"""P0 in-process smoke tests for the VisionMate API."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import app


def _make_black_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Return a solid black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


class _FakeVideoCapture:
    """Minimal cv2.VideoCapture stub returning a deterministic frame."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._opened = True
        self.width = kwargs.get("width", 640)
        self.height = kwargs.get("height", 480)

    def isOpened(self) -> bool:  # noqa: N802 (OpenCV naming)
        return self._opened

    def read(self) -> tuple[bool, np.ndarray]:
        return True, _make_black_frame(self.width, self.height)

    def release(self) -> None:
        self._opened = False

    def set(self, prop_id: int, value: float) -> bool:  # noqa: N802
        if prop_id == 3:
            self.width = int(value)
        elif prop_id == 4:
            self.height = int(value)
        return True


@contextmanager
def _patch_video_capture(monkeypatch: pytest.MonkeyPatch):
    import importlib

    module = importlib.import_module("cv2")
    monkeypatch.setattr(module, "VideoCapture", _FakeVideoCapture)
    yield


def _write_lidar_snapshot(path: pathlib.Path, timestamp: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": timestamp, "forward_min_distance": 2.3}
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(name="test_client")
def fixture_test_client(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> Iterator[TestClient]:
    capture_dir = tmp_path / "captured_images"
    run_dir = tmp_path / "runs"
    weights_dir = tmp_path / "weights"
    monkeypatch.setenv("VISIONMATE_CAPTURE_DIR", str(capture_dir))
    monkeypatch.setenv("VISIONMATE_RUN_DIR", str(run_dir))
    monkeypatch.setenv("VISIONMATE_WEIGHTS_DIR", str(weights_dir))
    monkeypatch.setenv("VISIONMATE_LIDAR_SNAPSHOT", str(tmp_path / "lidar" / "latest_lidar.json"))
    monkeypatch.setenv("VISIONMATE_LIDAR_TTL", "5")
    with _patch_video_capture(monkeypatch):
        with TestClient(app) as client:
            yield client


def _post_payload(client: TestClient, text: str, camera_index: int = 0) -> Dict[str, Any]:
    response = client.post(
        "/api/v1/mcp",
        json={"api_version": "v1", "text": text, "camera_index": camera_index},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["api_version"] == "v1"
    return payload


def test_api_contract_and_timings(test_client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    lidar_path = tmp_path / "lidar" / "latest_lidar.json"
    _write_lidar_snapshot(lidar_path, int(time.time()))

    payload = _post_payload(test_client, "前面有没有障碍物？")

    assert payload["status"] in {"ok", "warn"}
    assert isinstance(payload.get("message"), str)
    assert payload.get("data") is not None

    data_meta = payload["data"]["meta"]
    timings = data_meta.get("timings", {})
    assert "capture" in timings
    assert "capture_ms" in timings
    assert "total" in timings
    assert "total_ms" in timings

    meta = payload.get("meta", {})
    snapshot_meta = meta.get("lidar_snapshot")
    assert snapshot_meta is not None
    assert snapshot_meta.get("fresh") is True
    assert snapshot_meta.get("data", {}).get("forward_min_distance") == 2.3
    assert "snapshot_age_ms" in snapshot_meta


def test_lidar_expiry_behavior(test_client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    lidar_path = tmp_path / "lidar" / "latest_lidar.json"
    timestamp = int(time.time())
    _write_lidar_snapshot(lidar_path, timestamp)

    payload_fresh = _post_payload(test_client, "查询雷达状态")
    fresh_meta = payload_fresh.get("meta", {}).get("lidar_snapshot", {})
    assert fresh_meta.get("fresh") is True

    monkeypatch.setenv("VISIONMATE_LIDAR_TTL", "0.0")
    payload_stale = _post_payload(test_client, "再次查询雷达状态")
    stale_meta = payload_stale.get("meta", {}).get("lidar_snapshot", {})
    assert stale_meta.get("fresh") is False
    assert "data" not in stale_meta


def test_camera_fallback_attempts(test_client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    class _FailingVideoCapture(_FakeVideoCapture):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.invocations = 0

        def read(self) -> tuple[bool, np.ndarray]:
            self.invocations += 1
            if self.invocations == 1:
                return False, np.empty((0, 0, 3), dtype=np.uint8)
            return super().read()

    import importlib

    module = importlib.import_module("cv2")
    monkeypatch.setattr(module, "VideoCapture", _FailingVideoCapture)

    payload = _post_payload(test_client, "请切换分辨率试试看", camera_index=7)
    meta = payload.get("meta", {})
    attempts = meta.get("capture_attempts", [])
    assert attempts, "expected capture attempts to be recorded"
    assert any(item.get("success") for item in attempts), "should succeed after fallback"
