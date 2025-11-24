#!/usr/bin/env python3
"""Thin Whisper+TTS client that delegates intent routing to the MCP server."""
from __future__ import annotations

import os
import queue
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterator, List

import numpy as np
import torch
import requests
import sounddevice as sd
import pyttsx3
from faster_whisper import WhisperModel

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    SILERO_AVAILABLE = True
except Exception:
    SILERO_AVAILABLE = False

# === Configurations ===
MODEL_SIZE = "medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
LANGUAGE = "zh"

SAMPLE_RATE = 16000
BLOCK_MS = 30
SILENCE_TAIL_S = 0.3
MAX_BUFFER_S = 1.5
TRIGGER_S = 1.0
ENERGY_ABS_DBFS = -50.0
ENERGY_MARGIN_DB = 6.0
NOISE_WIN_S = 3.0
NOISE_FLOOR_INIT = -60.0
NO_SPEECH_PROB_TH = 0.90
AVG_LOGPROB_TH = -1.00
COMPRESSION_RATIO_TH = 2.40
API_URL = "http://127.0.0.1:9000/api/v1/mcp"

HALLUCINATIONS = [
    "点赞", "订阅", "转发", "打赏", "字幕", "频道", 
    "copyright", "amara", "观看", "视频", "支持", "明镜", "点点"
]

# Global Flag for Software AEC (Auto Echo Cancellation)
IS_SPEAKING = False

_audio_q: queue.Queue[np.ndarray] = queue.Queue()


def flush_audio_queue() -> None:
    """Empty the microphone buffer to drop TTS echoes."""
    with _audio_q.mutex:
        _audio_q.queue.clear()
    print("[DEBUG] Audio queue flushed.")

# Note: Removed global _engine to prevent blocking. We re-init per speak.

def speak(message: str) -> None:
    """TTS wrapper with Re-init strategy and Echo Cancellation."""
    global IS_SPEAKING
    if not message:
        return
    
    print(f"[TTS] Speaking: {message}")
    IS_SPEAKING = True # Mute Mic
    
    try:
        # Re-initialize engine every time to prevent 'runAndWait' freeze
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 0.9)
        engine.say(message)
        engine.runAndWait()
        engine.stop()
        del engine
    except Exception as e:
        print(f"[TTS ERROR] {e}")
    finally:
        # Small buffer to let echo fade
        time.sleep(0.5)
        flush_audio_queue()
        IS_SPEAKING = False # Unmute Mic


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    clipped = np.clip(samples, -1.0, 1.0)
    rms = float(np.sqrt(np.mean(clipped * clipped) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


class EnergyTracker:
    def __init__(self, block_ms: int = BLOCK_MS, win_s: float = NOISE_WIN_S, init_db: float = NOISE_FLOOR_INIT) -> None:
        self.maxlen = max(1, int(win_s * 1000 / block_ms))
        self.ring = deque([init_db] * self.maxlen, maxlen=self.maxlen)

    def update_block(self, block: np.ndarray) -> None:
        self.ring.append(rms_dbfs(block))

    def noise_floor_dbfs(self) -> float:
        arr = np.asarray(self.ring, dtype=np.float32)
        return float(np.percentile(arr, 20.0)) if arr.size else NOISE_FLOOR_INIT

    def gate_threshold_dbfs(self) -> float:
        return max(ENERGY_ABS_DBFS, self.noise_floor_dbfs() + ENERGY_MARGIN_DB)


def audio_callback(indata: np.ndarray, frames: int, t, status) -> None:
    if status:
        print("[WARN] Audio status", status)
    _audio_q.put(indata.copy().reshape(-1))


def mic_stream() -> Iterator[np.ndarray]:
    global IS_SPEAKING
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback):
        while True:
            block = _audio_q.get()
            
            # AEC: Drop audio packets while TTS is speaking
            if IS_SPEAKING:
                continue
                
            yield block


def vad_segmentation(stream: Iterator[np.ndarray], vad_model, energy_tracker: EnergyTracker) -> Iterator[np.ndarray]:
    buffer = np.zeros(0, dtype=np.float32)
    for block in stream:
        energy_tracker.update_block(block)
        buffer = np.concatenate([buffer, block])
        if len(buffer) < SAMPLE_RATE:
            continue
        if not SILERO_AVAILABLE or vad_model is None:
            yield buffer.copy()
            buffer = np.zeros(0, dtype=np.float32)
            continue
        speech_segments = get_speech_timestamps(buffer, vad_model, sampling_rate=SAMPLE_RATE)
        if speech_segments:
            last_end = speech_segments[-1]["end"]
            tail = (len(buffer) - last_end) / SAMPLE_RATE
            if tail > SILENCE_TAIL_S and last_end > 0.5 * SAMPLE_RATE:
                segment = buffer[:last_end]
                seg_dbfs = rms_dbfs(segment)
                if seg_dbfs >= energy_tracker.gate_threshold_dbfs():
                    yield segment.copy()
                buffer = buffer[last_end:]
        else:
            if len(buffer) > MAX_BUFFER_S * SAMPLE_RATE:
                buffer = np.zeros(0, dtype=np.float32)


def filter_segments(segments: List[dict]) -> List[dict]:
    filtered: List[dict] = []
    for seg in segments:
        if getattr(seg, "no_speech_prob", 0.0) > NO_SPEECH_PROB_TH:
            continue
        if getattr(seg, "avg_logprob", 0.0) < AVG_LOGPROB_TH:
            continue
        if getattr(seg, "compression_ratio", 0.0) > COMPRESSION_RATIO_TH:
            continue
        filtered.append(seg)
    return filtered


def forward_segment_to_mcp(text: str) -> None:
    payload = {"api_version": "v1", "text": text, "camera_index": None, "tts": False}
    try:
        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or data.get("intent", {}).get("msg")
        if message:
            print("[MCP]", message)
            speak(message)
    except requests.RequestException as exc:
        print(f"[ERROR] forwarding to MCP: {exc}")


def transcribe(model: WhisperModel, segment: np.ndarray) -> None:
    # Prevent transcription while TTS is active
    if IS_SPEAKING:
        return

    flush_audio_queue()

    audio = segment.astype("float32")
    segments_iter, _ = model.transcribe(
        audio,
        beam_size=5,
        language=LANGUAGE,
        initial_prompt="以下是中文、粤语和英语的混合语音指令。",
        vad_filter=True,
    )

    segments = list(segments_iter)
    segments = filter_segments(segments)

    for seg in segments:
        text = seg.text.strip()

        if not text:
            continue

        if any(keyword in text for keyword in HALLUCINATIONS):
            print(f"[IGNORED HALLUCINATION] {text}")
            continue

        if len(text) < 2:
            continue

        print(f"[TRANSCRIBED] {text}")
        forward_segment_to_mcp(text)


def add_cuda_paths_for_windows() -> None:
    if os.name != "nt":
        return
    venv = Path(sys.executable).parent.parent
    nvidia_base = venv / "Lib" / "site-packages" / "nvidia"
    candidates = [
        nvidia_base / "cuda_runtime" / "bin",
        nvidia_base / "cublas" / "bin",
        nvidia_base / "cudnn" / "bin",
    ]
    for path in candidates:
        if path.exists():
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(path))
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


def main() -> None:
    if os.name == "nt":
        add_cuda_paths_for_windows()
    print("[INFO] loading Whisper model...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    energy_tracker = EnergyTracker()
    vad_model = load_silero_vad() if SILERO_AVAILABLE else None
    print(f"[INFO] Ready. Running on {DEVICE}. Speak now.")

    for segment in vad_segmentation(mic_stream(), vad_model, energy_tracker):
        print("[LISTENING] Processing...")
        transcribe(model, segment)


if __name__ == "__main__":
    main()