from pathlib import Path
from typing import Any, Dict, Optional

import pytest

import orchestrator.router as router_module
from vision.engine import CVModuleProcess
from vision.backends import StubBackends


class RecordingBackends(StubBackends):
    """Record incoming kwargs so tests can assert fusion-side adjustments."""

    def __init__(self) -> None:
        super().__init__()
        self.last_kwargs: Optional[Dict[str, Any]] = None

    def check_hazards(self, image_path: str, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        self.last_kwargs = dict(kwargs)
        return {
            "status": "stub",
            "message": "stub hazard backend",
        }


def test_hazard_emergency_gate_sets_high() -> None:
    backend = RecordingBackends()
    engine = CVModuleProcess(backends=backend)

    payload = engine.run_task(
        "vision.check_hazards",
        "frame.jpg",
        {"forward_min_distance": 0.4},
    )

    assert payload["status"] == "ok"
    result = payload["result"]
    assert result is not None
    assert result.meta.get("hazard_level") == "high"
    assert pytest.approx(result.meta.get("forward_min_distance"), rel=1e-6) == 0.4


def test_hazard_adaptive_thresholds_relax() -> None:
    backend = RecordingBackends()
    engine = CVModuleProcess(backends=backend)

    payload = engine.run_task(
        "vision.check_hazards",
        "frame.jpg",
        {"forward_min_distance": 1.2},
    )

    assert payload["status"] == "ok"
    result = payload["result"]
    assert result is not None
    assert result.meta.get("hazard_level") == "medium"
    thresholds = backend.last_kwargs or {}
    assert pytest.approx(thresholds.get("yolo_conf_threshold"), rel=1e-6) == 0.3
    assert payload["meta"]["applied_thresholds"]["yolo_conf_threshold"] == pytest.approx(0.3, rel=1e-6)


def test_missing_lidar_snapshot_returns_fresh_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[Any, Any]] = []

    def fake_read(path: Optional[str] = None, ttl: Optional[float] = None) -> Optional[Dict[str, Any]]:
        calls.append((path, ttl))
        return None

    monkeypatch.setattr(router_module, "read_latest_snapshot", fake_read)
    monkeypatch.setattr(router_module, "load_model", lambda *_, **__: (object(), {}))
    monkeypatch.setattr(
        router_module,
        "route_mcp",
        lambda *_: {"action": "call", "function": "vision.check_hazards", "args": {}},
    )

    def fake_capture_image(*_args: Any, **_kwargs: Any) -> str:
        image_path = tmp_path / "frame.jpg"
        image_path.write_bytes(b"stub")
        return str(image_path)

    monkeypatch.setattr(router_module.frame_capture, "capture_image", fake_capture_image)

    snapshot_path = tmp_path / "lidar.json"
    monkeypatch.setenv("VISIONMATE_LIDAR_SNAPSHOT", str(snapshot_path))

    orchestrator = router_module.Orchestrator(capture_dir=str(tmp_path))
    orchestrator._cv = CVModuleProcess()

    response = orchestrator.handle_text("确认安全")

    assert response["status"] == "ok"
    lidar_meta = response["meta"].get("lidar_snapshot") or {}
    assert lidar_meta.get("fresh") is False
    assert lidar_meta.get("data") is None
    assert len(calls) == 1


def test_run_task_unsupported_function_warns() -> None:
    engine = CVModuleProcess()
    payload = engine.run_task("vision.unsupported", "frame.jpg", None)

    assert payload["status"] == "warn"
    assert payload["meta"].get("error_code") == "unsupported_function"
