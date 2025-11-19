# VisionMate 项目即时指南（2025-11-18）

## 1. 目标与架构概览

VisionMate 是一个面向视障/低视力用户的视觉辅助原型，核心流程是：

- **输入层**：麦克风 + ASR（`asr_realtime_whisper.py`）或 HTTP `/api/v1/mcp`；
- **编排层**：`orchestrator/router.py` 负责关键词/Sklearn 意图路由、摄像头回退、LiDAR 融合、调用 CV 引擎并生成结构化结果；
- **视觉层**：`vision/engine.py` + `vision/backends.py` 搭建可插拔的视觉管线（YOLOv8、EasyOCR、MiDaS 深度）和 LiDAR 协同，输出 `VisionResult`。


## 2. 快速上手

1. **准备环境**（Anaconda）：
   ```powershell
   conda env create -f environment.yml
   conda activate visionmate
   ```
2. **布置权重**：
   - 将 `yolov8n.pt` 放在 `weights/`；
   - `scripts/check_cameras.py` 检查摄像头索引；
   - 首次运行 EasyOCR/MiDaS 会联网下载模型，或使用 `weights/midas_v21_small_256.pt` 并设置 `VISIONMATE_MIDAS_WEIGHTS`。
3. **配置目录（可选）**：
   - `VISIONMATE_CAP_DIR`：保存图片的目录（默认 `captured_images/`）；
   - `VISIONMATE_RUN_DIR`：CV 引擎的运行目录（默认 `runs/`）；
4. **启动服务**：
   ```powershell
   uvicorn main:app --reload --port 9000
   客户端 可暂时用 python verify_cil_client.py 验证，后面写个语音输'asr_realtime_whisper.py'改改就行。
   目前 先执行写死的关键字匹配，若想看模型训练效果可以避开关键字输入
   ```
5. **验证与演示**：
   - 使用 `test_client.py` 发起 `/api/v1/mcp` 请求（示例 payload 可参见 `tests/test_p0_smoke.py`）；
   - 运行 `python scripts/run_keywords.py --sample-capture` 自动依次触发意图并观察 payload；
   - 通过 `python scripts/camera_preview.py` 保持摄像头常开并在屏幕上实时显示画面；
   - `python scripts/check_cameras.py` 可列出可用摄像头索引。

## 3. 目录映射（关键信息）

| 路径 | 说明 |
| --- | --- |
| `main.py` | FastAPI 入口，初始化 `Orchestrator` 和 `/api/v1/mcp`。 |
| `orchestrator/router.py` | 文本意图 → 画面采集 → CV 执行 → LiDAR 融合。 |
| `vision/engine.py` | 视觉任务的门户，统一输出 `VisionResult`、封装融合逻辑。 |
| `vision/backends.py` | 提供 `RealBackends`（YOLO + EasyOCR + MiDaS）。 |
| `vision/utils.py` | 文件夹/路径、相机索引、时间辅助工具。 |
| `frame_capture.py` | 抓帧库，封装分辨率 fallback + 多次重试。 |
| `lidar/lidar_bridge.py` | 从 `runs/lidar_snapshot.json` 读取 LiDAR 数据并判断 TTL。 |
| `scripts/*` | 包含意图训练/推理脚本与摄像头相关工具（`check_cameras.py`、`run_keywords.py`、`camera_preview.py`）。 |
| `tests/` | Pytest 套件，验证 API 流、水印、LiDAR 规则、深度元数据等。 |

## 4. 关键流程速览

1. **HTTP / API 流**：`main` → `Orchestrator.handle_text` → `frame_capture.capture_image` → `CVModuleProcess.run_task` → `VisionResult` → `schemas` 输出 `MCPResponse`。
2. **ASR Demo**：`asr_realtime_whisper.py` 捕获语音、转写、调用 `/api/v1/mcp` 并通过 TTS 播报结果。
3. **CV 任务**：`RealBackends` 以 `YOLO`, `EasyOCR`, `MiDaS` 实现 `detect`, `distance`, `describe`, `ocr`；`Engine` 中 `_run_check_hazards` 将 LiDAR/深度/YOLO 结果融合并注入 `meta`。

## 5. 
- **MiDaS 加载策略**：
  - 可通过 `VISIONMATE_MIDAS_MODEL` 覆盖模型名称，默认使用 `midas_small`；
  - 会自动检测 `weights/` 下的 `midas_*.pt`（例如 `midas_v21_small_256.pt`）并加载该 checkpoint；
  - 支持 `VISIONMATE_MIDAS_WEIGHTS` 或传参指定本地 `.pt`，便于在 timm 缺少模型时继续运行；
  - 加载本地权重时会打印路径并兼容不同 checkpoint key (`state_dict`、`model`)。 
- **持续预览脚本**：`scripts/camera_preview.py` 可选 `--camera` `--resolution` `--no-overlay` 参数，在 OpenCV 窗口保持摄像头流并叠加时间戳，按 `q` 关闭；非常适合在演示/调试过程中观察真实画面。
- **快速命令工具**：
  - `scripts/run_keywords.py` 自动调用多个关键词意图并打印 payload；
  - `scripts/check_cameras.py` 用来发现可用摄像头索引；
  - `frame_capture` 本体提供分辨率回退与重试日志，`Orchestrator` 会把 `capture_attempts` 注入 `meta`。

## 6. 测试与验证

- **主要命令**：
   - `pytest tests/test_p0_smoke.py`（端到端 API + LiDAR timing）；
   - `pytest tests/test_p1_lidar_soft_fusion.py`（hazard 规则、LiDAR meta）；
   - `pytest tests/test_p2_depth_metadata.py`（深度 meta + `RealBackends.relative_to_absolute_depth`）。
- **运行时监控**：`Orchestrator` 将 `timings`（intent/capture/cv/total）、`capture_attempts`、`lidar_snapshot`、`meta.depth_error` 等写入响应，便于调试性能与故障。

## 7. 多模态/硬件配置要点

- `VISIONMATE_CAMERA_INDEX` 控制默认摄像头，`Orchestrator` 逻辑会自动尝试 `canvas`：`high→medium→low`；
- `LiDAR`：默认从 `runs/lidar_snapshot.json` 读取`forward_min_distance`，自带 TTL（`lidar.bridge` 返回 `fresh` 标记）；
- `captured_images/` 存放从 `frame_capture` 获得的 JPG，可通过 `VISIONMATE_CAP_DIR` 改写；

- `frames` 相关命令都不会阻塞，因为 `scripts/run_keywords.py` 可用 `--sample-capture` 参数生成合成帧。

## 8. 未完成的技术债（需要优先关注）

1. **场景理解仍是占位**：`describe_scene` 仅靠检测框和简单位置分类，缺少语义层级或多句组合；建议未来考虑引入模组化文本或多模态融合。
2. **OCR 输出提示已变得真实**：`CVModuleProcess.ocr_read` 现在将 EasyOCR 识别出的 `text` 填入 `ocr_text` 并构建 `caption`（如 “读到的文字是：...”），引用流程中也会通过 `pyttsx3` 播报结果，因此这项“占位”问题已基本解决。
3. **深度估计可视化未完成**：MiDaS 结果没有落地 `depth_path`/`overlay_path`，没有输出深度图或危险色块；可在 `runs/depth/` 下生成可视化 PNG 并把路径注入 `VisionResult`。
4. **性能与评估缺失**：目前只有 timing 数值，缺少准确率/mAP/OCR latency 等指标；建议补充自动化 benchmark 或利用 `runs/` 记录每次响应时间。
5. **ASR → Vision → TTS 闭环已经具备**：`asr_realtime_whisper.py` 中 `forward_segment_to_mcp` 会调用 `/api/v1/mcp` 并拿到 `message`，随后通过 `pyttsx3` 播报，形成完整“语音进→视觉理解→语音出”的演示路径。
6. **配置与阈值硬编码**：LiDAR 阈值、YOLO 置信、MiDaS close threshold 等散落在代码，尚未集中到配置文件或环境变量（可考虑 `config.yaml` 或 `pydantic.BaseSettings`）。

## 9. 完整功能测试：
powershell -ExecutionPolicy Bypass -File scripts/run_full_test.ps1