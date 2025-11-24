# VisionMate 快速上手

VisionMate 是一套端到端的感知栈，提供 FastAPI MCP 接口、规则驱动的 Orchestrator、实时摄像头采集、LiDAR 融合以及多种调试/可视化工具。本指南重点介绍如何运行整个项目、可用的可视化方式、测试脚本以及 LiDAR 数据是如何接入的。

## 1. 环境准备

1. **推荐：使用 Conda 创建虚拟环境**
   ```powershell
   conda env create -f environment.yml
   conda activate visionmate
   ```
2. **若不使用 Conda，可直接安装依赖**
   ```powershell
   pip install -r requirements.txt
   ```
3. 请确保系统允许访问摄像头。可按需设置以下环境变量：
   - `VISIONMATE_CAPTURE_DIR`：自定义截图保存目录（默认 `captured_images/`）。
   - `VISIONMATE_RUN_DIR`：自定义运行产物目录（默认 `runs/`）。
   - `VISIONMATE_WEIGHTS_DIR`：YOLO/MiDaS 权重所在目录（默认 `weights/`）。

## 2. 核心运行方式

### 2.1 FastAPI MCP 服务（推荐入口）
启动 Orchestrator 驱动的 MCP API：
```powershell
uvicorn main:app --host 0.0.0.0 --port 9000
```
- 接口：`POST /api/v1/mcp`
- 请求体（`schemas.MCPRequest`）：`{"text": "描述周围", "camera_index": null}`
- 响应包含意图分类、调度状态以及 `VisionResultModel` 结果。


### 2.2 调度
使用 CLI 脚本模拟语音/文本指令：
```powershell
python debug_router_flow.py 
```
脚本会自动拍摄一帧、走完整个意图/视觉流程，并打印序列化后的 `VisionResult`。同时在 `runs/latest_status.json` 中落盘本次运行的摘要与可视化链接。
脚本测试：
```powershell
python scripts/verify_cli_client.py 
```
真实入口：
```powershell
python scripts/client_voice.py 
```
### 2.3 其他工具脚本
- `python test_client.py`：最小 MCP 客户端，可在终端手动发请求。
- `python scripts/depth_validation_test.py --image path/to/frame.jpg`：验证 MiDaS 深度估计与距离标定。

## 3. 可视化与调试资产

每次 CV 调用都会在 `runs/` 目录下写出：
- `runs/debug_yolo_input.jpg`：YOLOv8 检测可视化。
- `runs/depth/*.png`：深度 raw/overlay 图对。
- `runs/latest_status.json`：本次任务的概要、原始图片路径与可视化链接。

运行完任意指令后，可使用可视化脚本：
```powershell
python visualize_latest.py
```
脚本会读取 `latest_status.json`，加载原图、YOLO 叠加图与 MiDaS 热力图，并用 matplotlib 展示，同时附上意图、原始文本、hazard level 与 message。


### 4.1 Pytest 自动化
```powershell
pytest
```
`tests/` 目录覆盖了调度器辅助函数、意图路由以及 CV 工具函数。

### 4.2 深度校验脚本
`scripts/depth_validation_test.py` 会对 MiDaS 深度与参考距离/LiDAR 锚点进行对比，示例：
```powershell
python scripts/depth_validation_test.py --image captured_images/sample.jpg --lidar 1.2
```
输出包含分位数统计、融合后的场景距离，并可在需要时导出深度热力图供人工检查。

## 5. LiDAR 数据接入流程

1. **硬件驱动：** 启动串口版 LiDAR 驱动脚本：
   ```powershell
   python lidar/lidar_normal_serial_odo_switch.py --odom-method p2l
   ```
   - 自动识别串口，实时读取 RPLIDAR A1 数据，需要时可保存 CSV/PLY。
   - 每帧计算 `forward_min_distance`（±30° 区域）。
2. **快照共享：** 脚本持续写入 `runs/lidar_snapshot.json`，示例：
   ```json
   {
     "forward_min_distance": 0.74,
     "source": "hardware",
     "timestamp": "2025-11-24T10:00:00.123456"
   }
   ```
3. **Orchestrator 利用：** `Orchestrator.handle_text` 在每次拍照前都会读取该快照，如有值则把距离传给 CV 模块，用于深度标尺与危险判断。
4. **危险联动：** 若 LiDAR 判定距离小于 `HAZARD_DIST_THRESHOLD`，脚本会蜂鸣、可选触发 MCP 的 `check hazards`，并保持快照实时更新。

## 6. 常见启动组合

| 场景 | 命令 |
| --- | --- |
| 完整 MCP（API + 摄像头 + CV） | `conda activate visionmate` → `uvicorn main:app --host 0.0.0.0 --port 9000` |
| 纯文本/命令调度 | `python debug_router_flow.py --text "描述周围"` |
| LiDAR 融合模式 | `python lidar/lidar_normal_serial_odo_switch.py`（保持运行）+ 任意 MCP/文本指令 |
| 可视化面板刷新 | 任意指令后 → `python visualize_latest.py` |

## 7. 使用提示

- 在 Windows + PowerShell 环境运行时，确保摄像头未被其他程序占用。
- `runs/latest_status.json` 每次调用都会覆盖，如需历史记录请自行备份。
- YOLO/MiDaS 权重默认放在 `weights/`，若放置在其他位置，请设置 `VISIONMATE_WEIGHTS_DIR`。
- 若在无硬件的演示环境中，可在 `config.py` 中设置 `SIMULATION_MODE=1` 启用 Stub 后端。

