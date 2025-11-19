# asr_realtime_whisper.py
# -----------------------------------------
# 实时麦克风 → VAD 切段 → 能量门限(Energy Gate) → faster-whisper 转写
# - 默认使用外部 Silero-VAD（更稳）
# - 叠加能量门限降低噪声误触发（动态阈: max(ABS, 噪声地板+裕量)）
# - Segment 级“静音/幻听”过滤（no_speech_prob / avg_logprob / compression_ratio）
# - Windows 可选自动注入 CUDA/cuDNN DLL 路径
# -----------------------------------------

import os
import sys
import time
import queue
import threading
import joblib, json
from pathlib import Path
from collections import deque

import numpy as np
import sounddevice as sd
# 原版本未显式依赖 requests，用于示例 MCP 调用。
import requests
import pyttsx3

# 查看可用设备
print(sd.query_devices())
print("Default input device:", sd.default.device)

from faster_whisper import WhisperModel  # GPU 推理依赖 CTranslate2
# 只有在启用外部 VAD 时才需要导入
try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    SILERO_AVAILABLE = True
except Exception:
    SILERO_AVAILABLE = False


# =========================
# 可配置项（根据环境微调）
# =========================

# 选择使用哪种 VAD（推荐 True：外部 Silero-VAD 更稳、更可控）
USE_EXTERNAL_VAD = True

# Windows: 是否自动注入 CUDA/cuDNN DLL 路径（装在虚拟环境的 nvidia-* 包中）
enable_windows_cuda_path_fix = True

# ASR 模型与推理设置
MODEL_SIZE   = "large-v3"
DEVICE       = "cuda"        # "cuda" 或 "cpu"
COMPUTE_TYPE = "float16"     # "float16" | "int8_float16" | "int8" | "float32"
BEAM_SIZE    = 5
LANGUAGE     = None          # "yue"/"zh"/"en"/None(自动)

# 采集与分段
SAMPLE_RATE     = 16000
BLOCK_MS        = 30
SILENCE_TAIL_S  = 0.3        # 外部 VAD：句末静音阈；0.3~0.4s 低延迟更好
MAX_BUFFER_S    = 1.5        # 外部 VAD：无语音时的最大缓冲时长（防堆积）
TRIGGER_S       = 1.0        # 内置 VAD：聚合触发窗口（秒），越小延迟越低

# ===== 能量门限（核心） =====
# 注意：dBFS 以 0 dBFS 为满幅，-∞ 为完全静音；float32 输入约在 [-1, 1]
ENERGY_ABS_DBFS   = -50.0    # 绝对下限（低于此视为环境噪声）
ENERGY_MARGIN_DB  = 6.0      # 相对噪声地板的裕量（噪声地板+裕量）
NOISE_WIN_S       = 3.0      # 噪声地板观测窗（秒），建议 2~4s
NOISE_FLOOR_INIT  = -60.0    # 初始噪声地板（dBFS）

# Segment 级额外过滤阈值（可按需要调整/注释掉）
NO_SPEECH_PROB_TH     = 0.90     # segment.no_speech_prob > 0.90 → 丢弃
AVG_LOGPROB_TH        = -1.00    # segment.avg_logprob < -1.0  → 丢弃
COMPRESSION_RATIO_TH  = 2.40     # segment.compression_ratio > 2.4 → 丢弃


# =========================
# Windows: 注入 DLL 路径（可选）
# =========================

def add_cuda_paths_for_windows():
    """
    将 venv 内 nvidia 库的 bin 目录加入 DLL 搜索路径，以解决 cudnn_ops64_9.dll 等找不到的问题。
    需先在 venv 安装 nvidia-cudnn-cu12 / nvidia-cuda-runtime-cu12 / nvidia-cublas-cu12 等。
    """
    try:
        if os.name != "nt":
            return
        venv = Path(sys.executable).parent.parent
        nvidia_base = venv / "Lib" / "site-packages" / "nvidia"
        candidates = [
            nvidia_base / "cuda_runtime" / "bin",
            nvidia_base / "cublas"       / "bin",
            nvidia_base / "cudnn"        / "bin",
        ]
        for p in candidates:
            if p.exists():
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(p))
                os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:
        print(f"[WARN] add_cuda_paths_for_windows failed: {e}")


# =========================
# 能量计算 & 噪声地板跟踪
# =========================

def rms_dbfs(x: np.ndarray) -> float:
    """
    短时能量(RMS)→ dBFS
    输入 x: float32 [-1,1] 单声道
    """
    if x.size == 0:
        return -120.0
    x = np.clip(x, -1.0, 1.0)
    # 加少量epsilon防 log(0)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)

class EnergyTracker:
    """
    维护近 NOISE_WIN_S 秒的块能量,用20%分位估计噪声地板（对瞬时脉冲不敏感）
    """
    def __init__(self, block_ms=30, win_s=NOISE_WIN_S, init_db=NOISE_FLOOR_INIT):
        self.maxlen = max(1, int(win_s * 1000 / block_ms))
        self.ring = deque(maxlen=self.maxlen)
        # 用若干初值填充，避免开机瞬时门限过严/过松
        self.ring.extend([init_db] * self.maxlen)

    def update_block(self, block: np.ndarray):
        self.ring.append(rms_dbfs(block))

    def noise_floor_dbfs(self) -> float:
        if not self.ring:
            return NOISE_FLOOR_INIT
        arr = np.asarray(self.ring, dtype=np.float32)
        return float(np.percentile(arr, 20.0))  # 下分位近似环境噪声水平

    def gate_threshold_dbfs(self) -> float:
        # 动态门限 = max(绝对门限, 噪声地板 + 裕量)
        return max(ENERGY_ABS_DBFS, self.noise_floor_dbfs() + ENERGY_MARGIN_DB)


# =========================
# 音频采集
# =========================

audio_q = queue.Queue()

def audio_cb(indata, frames, t, status):
    if status:
        print(status)
    audio_q.put(indata.copy().reshape(-1))  # float32 mono

def mic_stream():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_cb):
        while True:
            blk = audio_q.get()
            yield blk


# =========================
# 外部 VAD 切段（Silero-VAD） + 能量门限
# =========================

def vad_segmentation(stream_iter, vad_model, energy_tracker: EnergyTracker):
    """
    将连续音频按静音切成句段；对候选段落做能量门限，低能量段直接丢弃。
    """
    buf = np.zeros(0, dtype=np.float32)

    for block in stream_iter:
        energy_tracker.update_block(block)  # 记录块能量
        buf = np.concatenate([buf, block])

        if len(buf) >= SAMPLE_RATE:  # 每秒检测一次
            ts = get_speech_timestamps(buf, vad_model, sampling_rate=SAMPLE_RATE)

            if ts:
                last_end = ts[-1]["end"]
                tail_sil = (len(buf) - last_end) / SAMPLE_RATE
                if tail_sil > SILENCE_TAIL_S and last_end > 0.5 * SAMPLE_RATE:
                    segment = buf[:last_end]
                    # ==== 能量门限：段落级 ====
                    seg_dbfs = rms_dbfs(segment)
                    if seg_dbfs >= energy_tracker.gate_threshold_dbfs():
                        yield segment.copy()
                    # ==========================
                    buf = buf[last_end:]
            else:
                if len(buf) > MAX_BUFFER_S * SAMPLE_RATE:
                    # 无语音长缓冲：虽然也可 yield 空段，这里直接清空
                    buf = np.zeros(0, dtype=np.float32)


# =========================
# Segment 级过滤（Whisper 输出）
# =========================

def filter_segments(segments):
    """
    对 faster-whisper 返回的 segments 逐条过滤；
    """
    out = []
    for s in segments:
        nsp = getattr(s, "no_speech_prob", None)
        if nsp is not None and nsp > NO_SPEECH_PROB_TH:
            continue
        if getattr(s, "avg_logprob", -10.0) < AVG_LOGPROB_TH:
            continue
        if getattr(s, "compression_ratio", 0.0) > COMPRESSION_RATIO_TH:
            continue
        out.append(s)
    return out


# =========================
# 主流程
# =========================

# import frame_capture
import scripts.infer_sklearn_intent as intent
mdl, TAU = intent.load_model()
MAP = intent.MAP
# =========================
# 将识别出的文本（及可选摄像头编号）发送到 MCP 接口，并输出其返回的语音或结果。
# =========================
# MCP_ENDPOINT = os.environ.get("VISIONMATE_MCP_URL", "http://127.0.0.1:9000/mcp")
# --- NEW VERSION ---
MCP_ENDPOINT = os.environ.get("VISIONMATE_MCP_URL", "http://127.0.0.1:9000/api/v1/mcp")


_tts_engine = None


def _get_tts_engine() -> pyttsx3.engine.Engine:
    """Lazily initialise a simple offline TTS engine for demo use.

    The intent is not to be production-grade, but to provide an end-to-end
    audio feedback path that matches the project requirements.
    """

    global _tts_engine
    if _tts_engine is None:
        _tts_engine = pyttsx3.init()
        try:
            _tts_engine.setProperty("rate", 180)
        except Exception:
            # 某些平台可能不支持设置属性，忽略即可。
            pass
    return _tts_engine


def speak(text: str) -> None:
    """Speak a short message via the local TTS engine."""

    if not text:
        return
    engine = _get_tts_engine()
    engine.say(text)
    engine.runAndWait()


# 原版本：未实现 HTTP 形式的 MCP 调用，而是保留了 frame_capture.JSON_decode 注释。
# 新版本（2025-11-11）：提供 forward_segment_to_mcp，直接调用 FastAPI 接口。
def forward_segment_to_mcp(text: str, camera_index: int | None = None) -> None:
    """Send a recognised utterance to the MCP FastAPI endpoint."""
    # payload = {"text": text}
    # --- NEW VERSION ---
    payload = {"api_version": "v1", "text": text}
    if camera_index is not None:
        payload["camera_index"] = camera_index
    try:
        resp = requests.post(MCP_ENDPOINT, json=payload, timeout=5)
        data = resp.json()
        # 优先播报 message 字段，其次回退到原始 JSON 文本。
        message = data.get("message")
        if isinstance(message, str) and message:
            print(f"[MCP] {message}")
            speak(message)
        else:
            text_repr = str(data)
            print(f"[MCP] {text_repr}")
            speak(text_repr)
    except requests.RequestException as exc:
        print(f"[MCP][WARN] 无法调用服务: {exc}")

def main():
    print("Loading VAD & Whisper (this may take a while on first run)...")

    if enable_windows_cuda_path_fix:
        add_cuda_paths_for_windows()
        print("Windows CUDA/cuDNN path fix applied.")

    # 载入外部 VAD（如开启）
    vad_model = None
    if USE_EXTERNAL_VAD:
        if not SILERO_AVAILABLE:
            raise RuntimeError("silero-vad 未安装，且 USE_EXTERNAL_VAD=True。请先 pip install silero-vad")
        vad_model = load_silero_vad()
        print("VAD model loaded.")

    # 能量跟踪器（无论用不用内置 VAD，都用它来估计噪声地板）
    energy_tracker = EnergyTracker(block_ms=BLOCK_MS, win_s=NOISE_WIN_S, init_db=NOISE_FLOOR_INIT)

    # 载入 Whisper
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    # 启动麦克风流
    stream_iter = mic_stream()

    # 外部 VAD：基于切段逐段转写
    if USE_EXTERNAL_VAD:
        for seg in vad_segmentation(stream_iter, vad_model, energy_tracker):
            # seg 为 float32 波形（16k mono），已经通过能量门限
            segments, info = model.transcribe(
                seg, language=LANGUAGE, beam_size=BEAM_SIZE, vad_filter=False
            )
            segments = list(segments)
            segments = filter_segments(segments)
            text = "".join(s.text for s in segments).strip()
            if text:
                lang = f"(lang={info.language}) " if getattr(info, "language", None) else ""
                print(f"[ASR]{lang}{text}")
                print(intent.route_mcp(text, mdl, TAU, MAP))
                # 原版本：这里保留了 frame_capture.JSON_decode 的示例注释。
                # frame_capture.JSON_decode(intent.route_mcp(text, mdl, TAU, MAP))
                # 新版本（2025-11-11）：调用 FastAPI MCP 服务。
                forward_segment_to_mcp(text)

    # 内置 VAD：对聚合窗口做能量判定
    else:
        agg_buf = np.zeros(0, dtype=np.float32)

        for block in stream_iter:
            energy_tracker.update_block(block)
            agg_buf = np.concatenate([agg_buf, block])

            if len(agg_buf) >= TRIGGER_S * SAMPLE_RATE:
                # ==== 能量门限（聚合块） ====
                if rms_dbfs(agg_buf) < energy_tracker.gate_threshold_dbfs():
                    agg_buf = np.zeros(0, dtype=np.float32)
                    continue
                # ============================
                segments, info = model.transcribe(
                    agg_buf, language=LANGUAGE, beam_size=BEAM_SIZE,
                    vad_filter=True, vad_parameters=dict(min_silence_duration_ms=int(SILENCE_TAIL_S * 1000))
                )
                segments = list(segments)
                segments = filter_segments(segments)
                text = "".join(s.text for s in segments).strip()
                if text:
                    lang = f"(lang={info.language}) " if getattr(info, "language", None) else ""
                    print(f"[ASR]{lang}{text}")
                    print(intent.route_mcp(text, mdl, TAU, MAP))
                    # 原版本：保留 frame_capture.JSON_decode 注释占位。
                    # frame_capture.JSON_decode(intent.route_mcp(text, mdl, TAU, MAP))
                    # 新版本（2025-11-11）：调用 FastAPI MCP 服务。
                    forward_segment_to_mcp(text)
                # 清空或保留少量尾部实现“滚动窗”
                agg_buf = np.zeros(0, dtype=np.float32)


if __name__ == "__main__":
    try:
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = 1
        main()
    except KeyboardInterrupt:
        print("\n[ASR] stopped by user.")
    except Exception as e:
        print(f"[ERROR] {e}")
