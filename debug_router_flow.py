import json
import re
from pprint import pprint
from typing import Any

try:
    from orchestrator.router import Orchestrator
except ImportError:  # pragma: no cover - fallback for alternative layout
    import importlib

    Orchestrator = importlib.import_module("router").Orchestrator


TEST_TEXTS = [
    "左边有什么。",
    "右边有什么。",
    "前面有什么。",
    "危险。",
]


def _serialize_result(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    try:
        objects = getattr(result, "objects", None)
        labels = [getattr(obj, "cls", None) for obj in objects] if objects is not None else None
    except Exception:  # pragma: no cover
        labels = None
    return {
        "task": getattr(result, "task", None),
        "caption": getattr(result, "caption", None),
        "objects": labels,
        "meta": getattr(result, "meta", {}),
    }


def _dump_response(text: str, response: dict[str, any]) -> None:
    status = response.get("status", "unknown")
    snippet = re.sub(r"[^\w]+", "_", text)[:16].strip("_") or "input"
    filename = f"debug_flow_{snippet}_{status}.json"
    serializable = {
        "input": text,
        "response": {
            k: v
            for k, v in response.items()
            if k != "result"
        },
    }
    serializable["response"]["result"] = _serialize_result(response.get("result"))
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, indent=2)


def main() -> None:
    orchestrator = Orchestrator()

    for text in TEST_TEXTS:
        print("\n" + "=" * 80)
        print(f"[INPUT] {text}")
        response = orchestrator.handle_text(text)

        status = response.get("status")
        message = response.get("message")
        intent = response.get("intent") or {}
        outer_meta = response.get("meta") or {}
        result = response.get("result")

        print(f"[STATUS] {status}")
        print(f"[MESSAGE] {message}")
        print("[INTENT_SPEC]:")
        pprint(intent)

        print("\n[OUTER META - KEYS]:", list(outer_meta.keys()))
        for key in [
            "timings",
            "hazard_level",
            "lidar_snapshot",
            "forward_min_distance",
            "distance_m",
            "scene_distance_m",
        ]:
            if key in outer_meta:
                print(f"  {key} = ")
                pprint(outer_meta[key])

        if result is None:
            print("\n[RESULT] None")
        else:
            print("\n[RESULT] VisionResult snapshot:")
            try:
                task = getattr(result, "task", None)
                caption = getattr(result, "caption", None)
                objects = getattr(result, "objects", None)
                result_meta = getattr(result, "meta", {}) or {}
            except Exception as exc:  # pragma: no cover
                print("  [ERROR reading VisionResult]", exc)
                task = caption = objects = result_meta = None

            print(f"  task    = {task}")
            print(f"  caption = {caption}")
            if objects is not None:
                try:
                    count = len(objects)
                except Exception:
                    count = "unknown"
                print(f"  objects = {count} 个")
                try:
                    labels = [getattr(obj, "cls", None) for obj in objects]
                    print(f"  object labels = {labels}")
                except Exception:
                    pass
            else:
                print("  objects = None")

            print("\n  [RESULT META - KEYS]:", list(result_meta.keys()))
            for key in [
                "stub",
                "backend",
                "depth_available",
                "depth_error",
                "forward_min_distance",
                "distance_m",
                "scene_distance_m",
                "lidar_snapshot",
            ]:
                if key in result_meta:
                    print(f"    {key} = ")
                    pprint(result_meta[key])

        _dump_response(text, response)


if __name__ == "__main__":
    main()
