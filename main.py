 # 原版本实现保留如下（逐行注释以便对比）：
 # This script is the first version of MCP_Server
 # which uses a local LLM to handle ASR context
 # now is replaced by a simple model  
 # If you want to use it, please modify the test_client.py accordingly
 # as well as system_prompt.txt
 # and make sure you have a local LLaMA2 server running
 # or change to use OpenAI API directly
 #
 # from fastapi import FastAPI, Request
 # from openai import OpenAI
 # import json
 # import frame_capture
 # import os
 #
 # # Use uvicorn to run the app
 # # python -m uvicorn main:app --reload --port 9000
 #
 # # Ensure save directory exists
 # frame_capture.make_save_dir()
 #
 # # Function mapping
 # function_map = {
 #     "low": frame_capture.capture_low_resolution,
 #     "medium": frame_capture.capture_medium_resolution,
 #     "high": frame_capture.capture_high_resolution
 # }
 #
 # # Define the path to the system prompt file
 # SYSTEM_PROMPT_FILE = "system_prompt.txt"
 #
 # # Global variable to store the system prompt
 # system_prompt = ""
 #
 # # Function to load system prompt from file
 # def load_system_prompt():
 #     global system_prompt
 #     if os.path.exists(SYSTEM_PROMPT_FILE):
 #         with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
 #             system_prompt = f.read()
 #     else:
 #         system_prompt = f"""return JSON format without change:
 #         {{
 #             "function": "system_prompt load failed, use default",
 #             "args": [none]
 #         }}
 #         """
 #
 # # Initialize FastAPI app
 # app = FastAPI()
 #
 # # Load system prompt during startup
 # @app.on_event("startup")
 # def startup_event():
 #     load_system_prompt()
 #     print("System prompt loaded.")
 #
 # # Example endpoint to return the system prompt
 # @app.get("/system_prompt")
 # def get_system_prompt():
 #     return {"system_prompt": system_prompt}
 #
 # # Initialize OpenAI client for local LLaMA2
 # client = OpenAI(
 #     base_url="http://localhost:8080/v1",  # local LLaMA2 server URL
 #     api_key="sk-local"  #  local server does not require a real key
 # )
 #
 # # Build messages for LLM
 # def build_messages(user_input: str):
 #    return [
 #        {"role": "system", "content": system_prompt},
 #        {"role": "user", "content": user_input}
 #    ]
 #
 # # MCP endpoint
 # @app.post("/mcp")
 # async def mcp_handler(request: Request):
 #     data = await request.json()
 #     user_input = data.get("query")
 #
 #     try:
 #         # Call LLM
 #         response = client.chat.completions.create(
 #             model="alpaca-zh-7b",
 #             messages=build_messages(user_input),
 #             temperature=0.1
 #         )
 #         llm_output = response.choices[0].message.content
 #         print("LLM 输出原文：", llm_output) # for DEBUG purpose only
 #
 #         # Analyze LLM output JSON
 #         command = json.loads(llm_output)
 #         func_name = command["function"]
 #         args = command["args"]
 #         print(func_name, args) # for DEBUG purpose only
 #
 #         # Execute corresponding function
 #         if func_name in function_map:
 #             result = function_map[func_name](*args)
 #             return {"result": result}
 #         else:
#             return {"error": f"未知函数：{func_name}"}
 #     except Exception as e:
 #         print("错误信息：", str(e))
 #         return {"error": f"解析或执行失败：{str(e)}"}

 # 新版本（2025-11-11）：基于调度器的 FastAPI MCP 服务入口。
 # """FastAPI entry point for the VisionMate MCP service."""
 #
 # from __future__ import annotations
 #
 # from typing import Any, Dict, Optional
 #
 # from fastapi import FastAPI, HTTPException, Request
 #
 # import frame_capture
 # from orchestrator import Orchestrator
 #
 # # Run with: uvicorn main:app --reload --port 9000 (inside the Conda environment)
 #
 # app = FastAPI()
 # _orchestrator: Optional[Orchestrator] = None
 #
 #
 # @app.on_event("startup")
 # def startup_event() -> None:
 #     """Initialise the orchestrator and ensure the capture directory exists."""
 #     global _orchestrator
 #     frame_capture.make_save_dir()
 #     _orchestrator = Orchestrator()
 #
 #
 # @app.post("/mcp")
 # async def mcp_handler(request: Request) -> Dict[str, Any]:
 #     if _orchestrator is None:
 #         raise HTTPException(status_code=503, detail="服务初始化尚未完成。")
 #
 #     payload = await request.json()
 #     text = payload.get("text")
 #     camera_index = payload.get("camera_index")
 #
 #     return _orchestrator.handle_text(text, camera_index=camera_index)

# --- NEW VERSION ---
"""FastAPI entry point for the VisionMate MCP service (API v1)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from config import SIMULATION_MODE
from orchestrator import Orchestrator
from schemas import MCPRequest, MCPResponse, VisionResultModel
from vision.utils import ensure_dir

# CAPTURE_DIR_ENV = "VISIONMATE_CAPTURE_DIR"
# --- NEW VERSION ---
CAPTURE_DIR_ENV = "VISIONMATE_CAPTURE_DIR"
CAPTURE_DIR_LEGACY_ENV = "VISIONMATE_CAP_DIR"
RUN_DIR_ENV = "VISIONMATE_RUN_DIR"
WEIGHTS_DIR_ENV = "VISIONMATE_WEIGHTS_DIR"

app = FastAPI(title="VisionMate MCP", version="1.0")
_orchestrator: Optional[Orchestrator] = None
_runtime_dirs: Dict[str, Path] = {}

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


@app.on_event("startup")
def startup_event() -> None:
    """Initialise orchestrator and required runtime directories."""
    global _orchestrator, _runtime_dirs
    capture_dir, run_dir, weights_dir = _prepare_runtime_dirs()
    mode_label = "SIMULATION" if SIMULATION_MODE else "REAL HARDWARE"
    print(f"🚀 VISIONMATE STARTING... MODE: {mode_label}")
    _runtime_dirs = {
        "capture": capture_dir,
        "runs": run_dir,
        "weights": weights_dir,
    }
    _orchestrator = Orchestrator(
        capture_dir=str(capture_dir),
        run_dir=str(run_dir),
        weights_dir=str(weights_dir),
    )


@app.post("/api/v1/mcp", response_model=MCPResponse)
async def mcp_handler(request_data: MCPRequest) -> JSONResponse:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="服务初始化尚未完成。")

    total_start = time.perf_counter()
    orchestration = _orchestrator.handle_text(
        request_data.text,
        camera_index=request_data.camera_index,
    )

    result = orchestration.get("result")
    data_model = (
        VisionResultModel.from_domain(result)
        if result is not None
        else None
    )

    meta = orchestration.get("meta", {}) or {}
    timings = meta.setdefault("timings", {})
    timings["total_api"] = time.perf_counter() - total_start
    meta.setdefault("api_version", request_data.api_version)
    meta["runtime_dirs"] = {
        "captured_images": str(_runtime_dirs["capture"]),
        "runs": str(_runtime_dirs["runs"]),
        "weights": str(_runtime_dirs["weights"]),
    }

    response = MCPResponse(
        api_version=request_data.api_version,
        status=orchestration.get("status", "error"),
        message=orchestration.get("message"),
        data=data_model,
        intent=orchestration.get("intent"),
        meta=meta,
    )

    # return JSONResponse(content=response.dict(exclude_none=True))
    # --- NEW VERSION ---
    if request_data.tts:
        message = orchestration.get("message") or (result.message if result else None)
        tts_path = _maybe_synthesize_tts(message)
        if tts_path:
            meta.setdefault("tts_audio_path", tts_path)
    return response


def _prepare_runtime_dirs() -> tuple[Path, Path, Path]:
    """Create runtime directories, respecting environment overrides."""
    # capture_dir = ensure_dir(os.getenv(CAPTURE_DIR_ENV, "captured_images"))
    # --- NEW VERSION ---
    preferred_capture = os.getenv(CAPTURE_DIR_ENV)
    legacy_capture = os.getenv(CAPTURE_DIR_LEGACY_ENV)
    capture_root = preferred_capture or legacy_capture or "captured_images"
    capture_dir = ensure_dir(capture_root)

    run_dir = ensure_dir(os.getenv(RUN_DIR_ENV, "runs"))
    weights_dir = ensure_dir(os.getenv(WEIGHTS_DIR_ENV, "weights"))
    return Path(capture_dir), Path(run_dir), Path(weights_dir)


def _maybe_synthesize_tts(message: Optional[str]) -> Optional[str]:
    if not message or pyttsx3 is None:
        return None
    runs_path = _runtime_dirs.get("runs")
    if runs_path is None:
        return None
    tts_dir = ensure_dir(runs_path / "tts")
    timestamp = int(time.time() * 1000)
    tts_path = tts_dir / f"tts_{timestamp}.mp3"
    engine = pyttsx3.init()
    engine.save_to_file(message, str(tts_path))
    engine.runAndWait()
    return str(tts_path)
