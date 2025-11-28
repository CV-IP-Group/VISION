# VisionMate 项目技术报告

版本：主分支（main）
日期：2025-11-26

---

## 摘要

VisionMate 是一个面向视障/盲人避障的多模态感知系统原型。系统通过摄像头（RGB）、LiDAR（RPLIDAR）和语音输入，结合本地模型（YOLOv8、MiDaS、EasyOCR）与规则/轻量意图分类器，实现环境感知、深度估计、障碍物检测与语音提示。项目以 FastAPI 提供 MCP（Multi-Modal Control Protocol）接口，Orchestrator 负责意图路由与任务编排，视觉推理封装在 `vision` 模块中，LiDAR 通过串口驱动持续写入快照以供融合。

---

## 目录与关键文件概览

- `main.py`：FastAPI 服务入口，API 为 `POST /api/v1/mcp`，构造 `Orchestrator` 并返回 `VisionResult` 序列化。
- `orchestrator/`：任务调度器，主实现为 `orchestrator/router.py`。
- `vision/`：视觉引擎与后端实现。
  - `vision/backends.py`：`RealBackends`（YOLO、MiDaS、EasyOCR）和 `StubBackends`；MiDaS 模型加载、深度后处理、归一化与深度叠加在此。
  - `vision/engine.py`：`CVModuleProcess`，对外暴露 `detect_objects`、`distance_query`、`describe_scene`、`ocr_read` 等接口。
  - `vision/types.py`：`Box`、`VisionResult` 数据结构。
  - `vision/utils.py`：辅助函数。
- `frame_capture.py`：跨平台摄像头采集、重试与保存逻辑（`low/medium/high` 分辨率策略）。
- `lidar/`：LiDAR 相关。
  - `lidar/lidar_normal_serial_odo_switch.py`：RPLIDAR 读取、快照写入、危险告警与里程计工具（自动端口识别）。
  - `lidar/lidar_bridge.py`：读取 `runs/lidar_snapshot.json` 的非阻塞接口。
- `scripts/`：若干工具脚本（意图训练/路由、导出深度图、模拟 LiDAR、ASR 客户端等），如 `scripts/infer_sklearn_intent.py`、`scripts/export_depth_maps.py`。
- `weights/`：模型权重目录（需要放置 YOLO 权重 `yolov8n.pt` 等）。
- `runs/`, `captured_images/`：运行时产物与拍摄的图片。
- `requirements.txt` / `environment.yml`：依赖清单与 conda 环境说明。

---

## 系统总体架构

1. 接入层（输入）
   - 语音：`scripts/client_voice.py`（使用 `faster-whisper`）或其它 ASR 将语音转为文本并调用 MCP。
   - 摄像头：`frame_capture.py` 提供分辨率选择与保存机制，产生 JPEG 文件和 `runs/latest_rgb.jpg`。
   - LiDAR：`lidar/lidar_normal_serial_odo_switch.py` 读取串口，计算 `forward_min_distance`，写入 `runs/lidar_snapshot.json`。

2. 控制层（Orchestrator）
   - 入口：`orchestrator/router.py` 的 `Orchestrator.handle_text`。
   - 意图识别：规则优先（`scripts/infer_sklearn_intent.py` 中 `_RULES`），回退到 sklearn 模型（`models/intent_sklearn.joblib`）。
   - 流水线：采集图片 → 调用 `CVModuleProcess.run_task` → 深度/LiDAR 融合 → 生成 `VisionResult`。

3. 感知层（视觉引擎）
   - `vision/engine.py` 负责把后端能力封装为统一接口。
   - `vision/backends.RealBackends` 管理 YOLO、EasyOCR、MiDaS 的加载与推理（支持本地权重、`torch.hub`、`timm`）。

4. 输出层
   - MCP API 返回 JSON（`schemas.MCPResponse`），可选本地 TTS（`pyttsx3`）写入 `runs/tts/`。
   - 可视化工具：`visualize_latest.py` 读取 `runs/latest_status.json` 并展示图像与深度叠加。

数据流动说明：输入（ASR 文本 + 摄像头 + LiDAR 快照）→ Orchestrator 路由 → CV 推理（YOLO/MiDaS/OCR）→ 深度融合（视觉深度 + LiDAR 校准）→ 结果注入并返回/播报。

---

## 多模态支持与实现细节

1. 摄像头（RGB）
   - 代码位置：`frame_capture.py`。
   - 分辨率：`low` (640×480)、`medium` (1280×720)、`high` (1920×1080)。
   - 预处理：重试、帧暖机、亮度检测（防过暗）。结果写入 `captured_images/` 与 `runs/latest_rgb.jpg`。

2. LiDAR（RPLIDAR）
   - 代码位置：`lidar/lidar_normal_serial_odo_switch.py`。
   - 输出快照：`runs/lidar_snapshot.json`，包含 `forward_min_distance`、`source`、`timestamp`。
   - Orchestrator 通过 `lidar/lidar_bridge.read_latest_snapshot()` 非阻塞读取此快照，TTL 默认为 3 秒（由 `VISIONMATE_LIDAR_TTL` 控制）。

3. 语音/ASR
   - 脚本：`scripts/client_voice.py`（`faster-whisper`）、`asr_realtime_whisper.py`。
   - 流程：实时识别 → 得到文本 → POST 到 `POST /api/v1/mcp`。

4. 深度估计（MiDaS）
   - 加载位置：`vision/backends.py` 中 `_setup_midas`, `_load_from_torch_hub`, `_load_local_midas`。
   - 模型切换：可通过构造参数 `midas_model`、环境变量 `VISIONMATE_MIDAS_MODEL` 或在 `weights/` 中放置 `midas_*.pt` 来指定。本地权重路径可用 `VISIONMATE_MIDAS_WEIGHTS` 覆盖。
   - 转换：`_apply_midas_transform` 兼容不同 transform 返回类型（tensor 或 dict）。

模态对齐策略：采用弱同步（文件/快照共享），适合演示与交互式体验，但非严格时间同步。

---

## 模型与算法详细说明

1. 目标检测：YOLOv8n
   - 加载：`weights/yolov8n.pt`，通过 `ultralytics.YOLO` 调用。
   - 输出：检测框转换为 `vision.types.Box`，随后可通过 `RealBackends.attach_depth_to_boxes` 注入深度。

2. 文本识别：EasyOCR
   - 初始化使用 `easyocr.Reader(['ch_sim','en'], gpu=torch.cuda.is_available())`。

3. 深度估计：MiDaS（MiDaS_small / DPT 等变体）
   - 支持来源：本地权重 / `torch.hub` / `timm`。
   - 后处理：插值回图像分辨率，`linear_depth_projection` 将相对深度映射到近/远范围（默认 0.5–8.0m），`calibrated_absolute_depth` 用 LiDAR 做比例缩放并做一致性检查。

4. 融合与决策
   - 融合方式：后期融合（结果级），靠 LiDAR 的 `forward_min_distance` 对视觉深度进行线性缩放。
   - 危险评估：`CVModuleProcess._handle_check_hazards` 基于近距离 box.dist 或视觉最小深度决定 `hazard_level`（safe/warning/danger），并生成文本提示。

5. 意图识别
   - 规则优先（`scripts/infer_sklearn_intent.py._RULES`），回退到 sklearn 模型（Tfidf + LogisticRegression + CalibratedClassifierCV）训练脚本见 `scripts/train_sklearn_intent.py`。

---

## 数据与训练

- 意图数据：`data/train.csv`, `data/dev.csv`, `data/test.csv`（文本 + 标签），规模为示例/小样本，适用于教学/演示。
- YOLO 与 MiDaS 权重：未包含训练流水线，权重需用户准备并放入 `weights/`。
- 训练脚本：`scripts/train_sklearn_intent.py`（生成 `models/intent_sklearn.joblib`）。

评估与可视化：项目提供 `visualize_latest.py` 和 `scripts/export_depth_maps.py` 用于人工检查深度估计与叠加图像。

---

## 运行与部署说明（快速）

前置环境：Python 3.11，建议使用 Conda 环境（`environment.yml`）。GPU 支持依赖正确安装的 NVIDIA 驱动与 CUDA 版本，以及含 CUDA 支持的 PyTorch。检查 CUDA 可用性：

```powershell
conda activate visionmate
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device count:', torch.cuda.device_count())
    print('device 0:', torch.cuda.get_device_name(0))
PY
```

启动 MCP 服务：
```powershell
uvicorn main:app --host 0.0.0.0 --port 9000
```

本地测试示例：
```powershell
python debug_router_flow.py
python scripts/export_depth_maps.py --image captured_images/test.jpg --lidar-distance 2.0
```

若遇到 `ModuleNotFoundError: No module named 'vision'`，请在项目根目录运行或设置 `PYTHONPATH`：
```powershell
$env:PYTHONPATH = "D:\Vision"
python scripts/export_depth_maps.py --image captured_images/test.jpg --lidar-distance 2.0
```

---

## 安全性、鲁棒性与局限性

- 本项目的深度转换与 LiDAR 校准为启发式方法，不适用于严格安全决策（如自动紧急制动）。代码中多处注释提示“非安全级别”。
- 鲁棒性增强：摄像头捕获有暖机与亮度检查，LiDAR 快照有 TTL 校验，MiDaS 校准有异常检测逻辑（有效点阈值、视觉与 LiDAR 差异上限）。
- 隐私：默认本地运行，项目并不强制上传图片或传感器数据到第三方服务。若开启网络请求（如 MCP 远程调用），需注意隐私策略。

---

## 建议改进（优先级排序）

1. 自动化权重管理脚本：下载/校验 `yolov8n.pt`、MiDaS 预训练权重，减少手工准备步骤。
2. 增强时间同步：实现摄像头与 LiDAR 的时间戳对齐或外参标定流程以提升深度融合精度。
3. 对 MiDaS 深度估计引入外参/相机内参计算（如果可获得）以替代启发式线性映射。
4. 扩展评估流水线：记录推理延迟 / 精度指标，增加 mAP/召回等感知指标的离线评估脚本。
5. 提供 Dockerfile/compose 与轻量化模型（ONNX/ TensorRT）以便部署到边缘设备。

---

## 参考文件索引（便于审阅）

- API：`main.py`
- Orchestrator：`orchestrator/router.py`
- 视觉引擎：`vision/engine.py`, `vision/backends.py`, `vision/types.py`, `vision/utils.py`
- 摄像头：`frame_capture.py`
- LiDAR：`lidar/lidar_normal_serial_odo_switch.py`, `lidar/lidar_bridge.py`
- 脚本：`scripts/export_depth_maps.py`, `scripts/infer_sklearn_intent.py`, `scripts/train_sklearn_intent.py`, `scripts/client_voice.py`
- 依赖：`requirements.txt`, `environment.yml`



END
