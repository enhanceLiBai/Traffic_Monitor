# 🚗 智能交通监测系统 — 训练与部署说明

---

## 📋 目录

1. [系统架构概览](#1-系统架构概览)
2. [环境搭建](#2-环境搭建)
3. [数据集准备](#3-数据集准备)
4. [模型训练](#4-模型训练)
5. [本地部署运行](#5-本地部署运行)
6. [远程服务器部署（AutoDL）](#6-远程服务器部署autodl)
7. [常见问题排查](#7-常见问题排查)
8. [Linux 生产环境部署（systemd）](#8-linux-生产环境部署systemd-服务)

---

## 1. 系统架构概览

本系统采用**三级级联架构**，由三个独立模型协同工作：

```
视频输入 (traffic.mp4)
    │
    ▼
┌──────────────────────────────────────┐
│  Stage 0: YOLOv8m (车辆检测+追踪)      │
│  模型: yolov8m.pt                     │
│  功能: 检测车辆(轿车/客车/卡车)         │
│        ByteTrack 多目标追踪分配 ID      │
└──────────────────────────────────────┘
    │ 车辆中心点越过虚拟激光线
    ▼
┌──────────────────────────────────────┐
│  Stage 1: 自训练 YOLO (车牌定位)       │
│  模型: best.pt (自训练)               │
│  功能: 在4K原图上精准抠出车牌区域        │
└──────────────────────────────────────┘
    │ 车牌裁剪图
    ▼
┌──────────────────────────────────────┐
│  Stage 2: HyperLPR3 (字符识别)        │
│  引擎: hyperlpr3.LicensePlateCatcher  │
│  功能: 高置信度车牌字符OCR识别          │
└──────────────────────────────────────┘
    │
    ▼
  输出: 中文渲染视频 + CSV统计报表
```

**关键文件说明：**

| 文件 | 用途 |
|---|---|
| `traffic_monitor.py` | 🚀 系统主程序，一键运行 |
| `setup_roi.py` | 激光线配置 + 远程服务器一键部署 |
| `setup_roi_local.py` | 激光线配置 + 本地一键部署运行 |
| `dataset_tools/train.py` | YOLO 车牌检测模型训练脚本 |
| `dataset_tools/ccpd_to_yolo.py` | CCPD 数据集 → YOLO 格式转换 |
| `dataset_tools/split_dataset.py` | 数据集 8:1:1 划分 |
| `data.yaml` | 训练数据路径与类别配置 |
| `config.json` | 视频路径与激光线坐标配置 |
| `best.onnx` | 自训练车牌检测模型（ONNX 格式，用于推理加速） |
| `yolov8m.pt` | YOLOv8m 车辆检测预训练权重 |

---

## 2. 环境搭建

### 2.1 硬件要求

| 配置等级 | 硬件规格 | 预期性能 |
|---|---|---|
| **最低配置** | Intel i5 / Ryzen 5, 8GB RAM, 无独显 | 2~5 FPS (CPU推理) |
| **推荐配置** | Intel i7 / Ryzen 7, 16GB RAM, NVIDIA 独显 ≥4GB 显存 | 25~50+ FPS (CUDA加速) |
| **云端算力** | AutoDL 等平台租用 2080Ti / 3060 | 25~50+ FPS |

### 2.2 软件环境

- **操作系统**: Windows 10/11 或 Linux (Ubuntu 20.04+)
- **Python 版本**: **3.10**（强烈建议，3.8 已过期）
- **CUDA**: 若使用 NVIDIA 显卡，需安装 CUDA 11.8+ 及对应 cuDNN

### 2.3 一键安装依赖

```bash
# 在项目根目录执行（使用清华镜像源加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

依赖清单 (`requirements.txt`)：

```
hyperlpr3==0.1.3          # 车牌字符识别引擎
lap>=0.5.12               # ByteTrack 线性分配库（追踪必需）
numpy>=1.24.0             # 矩阵运算库
onnx>=1.15.0              # ONNX 模型支持（best.onnx 推理需要）
onnxruntime-gpu>=1.18.0   # ONNX Runtime GPU 推理引擎（CPU 环境改 onnxruntime）
opencv-python>=4.10.0     # 计算机视觉（含GUI，setup_roi.py 需要）
paramiko>=3.0.0           # SSH/SFTP 远程部署（setup_roi.py 需要）
Pillow>=10.0.0            # 中文抗锯齿字体渲染
ultralytics>=8.4.0        # YOLOv8 训练与推理框架
```

> ⚠️ **注意**：不要同时安装 `opencv-python`、`opencv-python-headless`、`opencv-contrib-python`，它们都提供 `cv2` 模块，互相覆盖会导致不可预知的错误。本系统只需 `opencv-python`（支持 GUI，setup_roi.py 画线工具依赖 `cv2.imshow`）。

### 2.4 Linux 系统中文字体补丁 ⚠️

Linux 容器/server 默认无中文字体，**必须执行**：

```bash
apt-get update && apt-get install -y fonts-wqy-zenhei
```

程序启动时自动探测字体路径（按 Wqy-Zenhei → Noto CJK → 系统字体顺序），无需手动配置 `FONT_PATH`。

> Windows 系统内置中文字体（SimHei / 微软雅黑），程序自动识别，无需此步骤。

### 2.5 CUDA 环境安装指南（GPU 用户）

使用 NVIDIA 显卡进行训练/推理需安装 CUDA 工具包。

**Step 1 — 验证显卡驱动：**
```bash
nvidia-smi
```
确保驱动版本 ≥ 525.x，输出中 `CUDA Version` 显示 11.8 或更高。

**Step 2 — 版本兼容性对照表：**

| PyTorch 版本 | CUDA 版本 | cuDNN |
|---|---|---|
| 2.0.x | 11.8 | 8.7+ |
| 2.1.x | 12.1 | 8.9+ |
| 2.2.x+ | 12.1 / 12.4 | 9.0+ |

> `ultralytics` 依赖 PyTorch，安装时会自动拉取匹配的 CUDA 版本，通常无需手动安装 CUDA Toolkit。若 `nvidia-smi` 正常但 `torch.cuda.is_available()` 返回 `False`，请根据 PyTorch 官网指南重新安装对应 CUDA 版本的 PyTorch。

**Step 3 — 验证 GPU 可用性：**
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### 2.6 依赖分环境说明

`requirements.txt` 默认包含 `onnxruntime-gpu`（GPU 推理）。CPU 用户请手动替换：

```bash
# CPU 环境：先卸载 GPU 版，再安装 CPU 版
pip uninstall onnxruntime-gpu -y
pip install onnxruntime>=1.18.0
```

---

## 3. 数据集准备

本项目的车牌检测模型使用 **CCPD (Chinese City Parking Dataset)** 进行训练。

### 3.1 数据集下载

- **CCPD 中文车牌数据集**: [百度 AI Studio](https://aistudio.baidu.com/datasetdetail/101620)
- 原始项目: [GitHub - detectRecog/CCPD](https://github.com/detectRecog/CCPD)
- 推荐使用 `ccpd_base` 子集（约 20 万张），在精度与训练时间之间取得较好平衡

### 3.2 数据集划分 (8:1:1)

```bash
# 1. 修改 split_dataset.py 中的路径
#    src_dir = "你的CCPD数据集实际路径"
#    output_base = "./data"

# 2. 执行划分
python dataset_tools/split_dataset.py
```

**划分策略**（固定随机种子 `seed=42`）：

| 集合 | 比例 | 用途 |
|---|---|---|
| train/ | 80% | 模型训练 |
| val/ | 10% | 每轮验证，监控过拟合 |
| test/ | 10% | 最终测试评估 |

### 3.3 CCPD → YOLO 格式转换

CCPD 数据集的车牌坐标信息编码在**文件名**中，需要解析并转换为 YOLO 归一化格式。

**文件名解析规则：**

```
示例文件名: 025-91_95-131&428_599&571-...
                      └──────┬──────┘
                     第3段: "131&428_599&571"
                     解析: xmin=131, ymin=428, xmax=599, ymax=571
```

**YOLO 标签格式**（每张图片对应一个 `.txt` 文件）：

```
<class_id> <x_center> <y_center> <width> <height>
```

所有值均为归一化到 [0, 1] 的浮点数。车牌检测为单类任务，class_id 恒为 0。

```bash
# 1. 修改 ccpd_to_yolo.py 中的 base_path
#    base_path = "./data"  # 指向划分后的数据目录

# 2. 执行转换（自动处理 train/val/test 三个子集）
python dataset_tools/ccpd_to_yolo.py
```

### 3.4 验证数据集配置

转换完成后，确认 `data.yaml` 中的路径正确：

```yaml
train: ./data/train    # 训练集图片路径
val: ./data/val        # 验证集图片路径
test: ./data/test      # 测试集图片路径
nc: 1                  # 类别数（仅"车牌"一类）
names:
  0: license_plate     # 类别名称
```

---

## 4. 模型训练

### 4.1 训练策略

采用**迁移学习**策略：
- **基础权重**: YOLOv8m (`yolov8m.pt`)，已在 COCO 数据集上预训练
- **微调目标**: 车牌定位（单类别检测）
- **输入尺寸**: 640×640

### 4.2 训练参数

`dataset_tools/train.py` 中的默认配置：

```python
model.train(
    data="data.yaml",        # 数据集配置文件（相对路径）
    epochs=100,              # 训练轮数（50~100 轮通常收敛）
    imgsz=640,               # 输入尺寸
    batch=16,                # 批大小（大显存卡可调至 32/64）
    workers=8,               # 数据加载线程数
    device=0,                # GPU 编号（CPU 训练设为 "cpu"）
    optimizer="SGD",         # 优化器（可选 "AdamW"）
    project="runs",          # 输出根目录
    name="flow_plate_m",     # 实验名称
    plots=True,              # 生成训练曲线图
)
```

### 4.3 启动训练

```bash
# 1. 确保 yolov8m.pt 在项目根目录（首次运行会自动下载）
# 2. 确认 data.yaml 中的路径正确（默认为 ./data/train, ./data/val, ./data/test）
# 3. 启动训练
python dataset_tools/train.py
```

### 4.4 训练输出

训练结束后，在 `runs/flow_plate_m/` 下生成：

```
runs/flow_plate_m/
├── weights/
│   ├── best.pt          # 🌟 验证集上 mAP 最高的权重（部署用这个）
│   └── last.pt          # 最后一轮的权重（用于断点续训）
├── results.csv          # 每轮训练的 Loss 和 mAP 数据
├── confusion_matrix.png # 混淆矩阵
├── PR_curve.png         # P-R 曲线
└── results.png          # 训练过程曲线（Loss + mAP）
```

### 4.5 训练效果评估

| 指标 | 说明 | 优秀标准 |
|---|---|---|
| mAP@0.5 | IoU=0.5 时的平均精度 | > 0.95 |
| mAP@0.5:0.95 | IoU 从 0.5 到 0.95 的平均精度 | > 0.80 |
| Precision | 精确率 | > 0.95 |
| Recall | 召回率 | > 0.90 |

### 4.6 ONNX 模型导出（推理加速）⚡

训练得到的 `best.pt` 可直接用于推理，但导出为 ONNX 格式可显著提升推理速度（通常加速 20%~40%）。

```bash
# 在项目根目录执行
python -c "
from ultralytics import YOLO
model = YOLO('./runs/flow_plate_m/weights/best.pt')
model.export(format='onnx', imgsz=640, simplify=True)
print('✅ ONNX 导出完成: best.onnx')
"
```

导出后的 `best.onnx` 放到项目根目录即可被 `traffic_monitor.py` 自动加载。

| 格式 | 文件大小 | 推理速度 | 适用场景 |
|---|---|---|---|
| `best.pt` | ~20 MB | 基准 | 训练、验证、灵活性高 |
| `best.onnx` | ~10 MB | 快 20%~40% | 生产推理部署 |

> 💡 如果要在 `traffic_monitor.py` 中切换回 `.pt` 格式，修改 `my_plate_detector = YOLO("./runs/flow_plate_m/weights/best.pt")` 即可。

### 4.7 关键超参数调优建议

| 场景 | 建议 |
|---|---|
| 显存不足 (OOM) | 减小 `batch` 至 8 或 4 |
| 训练不收敛 | 降低学习率，换用 `optimizer="AdamW"` |
| 过拟合 | 增加数据增强 (`mosaic=1.0`, `flipud=0.5`) |
| 小目标漏检 | 提高 `imgsz` 至 1280 |

---

## 5. 本地部署运行

### 5.1 前置条件检查

- [ ] `yolov8m.pt` 已下载（首次运行自动拉取）
- [ ] `runs/flow_plate_m/weights/best.pt` 存在（自训练权重）
- [ ] `data/traffic.mp4` 测试视频存在
- [ ] 中文字体可用（Linux 需安装 `fonts-wqy-zenhei`）

### 5.2 方式一：直接运行主程序

```bash
python traffic_monitor.py
```

程序将自动完成以下步骤：
1. 加载 YOLOv8m 车辆检测模型
2. 加载自训练车牌检测模型
3. 初始化 HyperLPR3 识别引擎
4. 逐帧处理视频，实时追踪车辆
5. 车辆过线时触发抓拍 → 车牌定位 → OCR 识别
6. 输出中文渲染结果视频 `yolo_hy_plate_result.mp4`
7. 生成 CSV 报表 `recognition_results.csv`
8. 终端打印识别成功率统计

### 5.3 方式二：通过 ROI 配置工具运行（本地）

```bash
python setup_roi_local.py
```

**交互流程：**

1. 弹出文件选择窗口 → 选择待检测视频
2. 在视频首帧画面上**依次点击**：
   - 左端点 A（激光线起点）
   - 右端点 B（激光线终点）
3. 按 `Enter` 确认划线，系统保存配置到 `config.json`
4. 自动启动 `traffic_monitor.py` 进行检测

> 💡 该脚本适合**纯本地运行**场景，不涉及任何远程连接。

### 5.4 方式三：通过 ROI 配置工具运行（远程）

```bash
python setup_roi.py
```

**交互流程：**

1. 弹出文件选择窗口 → 选择待检测视频
2. 在视频首帧画面上**依次点击**：
   - 左端点 A（激光线起点）
   - 右端点 B（激光线终点）
3. 系统自动保存配置到 `config.json`
4. （可选）自动上传视频和配置到远程服务器

> 详见第 6 节「远程服务器部署」。

### 5.5 config.json 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `video_path` | string | 待检测视频的绝对路径 |
| `scale` | float | 画面缩放比例（0.5 = 缩小一半），降低计算量同时提升 FPS |
| `line_a` | [int, int] | 激光线起点坐标（基于缩放后画面，由 setup_roi 工具生成） |
| `line_b` | [int, int] | 激光线终点坐标（同上） |
| `is_configured` | bool | 配置完成标志，`true` 表示已划线配置 |

### 5.6 关键参数修改指南

在 `traffic_monitor.py` 中可直接修改：

```python
# 视频输入（也可通过 config.json 设置）
video_path = "./data/traffic.mp4"   # 改为你的视频路径

# 画面缩放（0.5 = 缩小一半，降低计算量）
SCALE = 0.5

# 虚拟激光线端点坐标（需要根据你的视频画面调整）
LINE_A = (int(610 * SCALE), int(1062 * SCALE))
LINE_B = (int(3317 * SCALE), int(999 * SCALE))

# 过线触发距离阈值（像素）
max_dist=15

# 追踪目标类别（2=轿车, 5=客车, 7=卡车）
ALLOWED_CLASSES = [2, 5, 7]
```

### 5.7 输出文件说明

| 输出文件 | 说明 |
|---|---|
| `yolo_hy_plate_result.mp4` | 中文车牌渲染结果视频 |
| `recognition_results.csv` | 车牌识别记录表（UTF-8-SIG 编码，Excel 可直接打开） |
| `debug_capture/` | 抓拍调试图片（SUCCESS=成功, FAILED=失败） |

---

## 6. 远程服务器部署（AutoDL）

### 6.1 适用场景

- 本地无 NVIDIA 独显，需要云端 GPU 推理
- 处理大批量视频，需要更高吞吐量

### 6.2 服务器环境准备

在 AutoDL 实例中执行：

```bash
# 1. 克隆/上传项目到 /root/autodl-tmp/FlowPlate-Analyzer/

# 2. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 安装中文字体（关键！）
apt-get update && apt-get install -y fonts-wqy-zenhei

# 4. 安装 SSH 服务（确保远程连接可用）
apt-get install -y openssh-server
```

### 6.3 配置 setup_roi.py

修改 `setup_roi.py` 中的服务器配置：

```python
SERVER_IP = "connect.nmb1.seetacloud.com"   # AutoDL 实例 SSH 地址
SERVER_PORT = 10743                          # SSH 端口
SERVER_USER = "root"                         # SSH 用户名
SERVER_PASS = "你的密码"                      # SSH 密码

SERVER_PROJECT_DIR = "/root/autodl-tmp/FlowPlate-Analyzer"  # 项目根目录
```

> ⚠️ **安全警告**：`SERVER_PASS` 以明文存储，**切勿**将含真实密码的 `setup_roi.py` 提交到 Git 仓库。推荐使用环境变量替代：
> ```python
> import os
> SERVER_PASS = os.environ.get("AUTODL_PASS", "")
> ```
> 并在终端中设置：`export AUTODL_PASS="你的密码"`（Linux）/ `$env:AUTODL_PASS="你的密码"`（PowerShell）。

### 6.4 一键部署流程

```bash
# 本地执行
python setup_roi.py
```

**自动化流程：**

1. 📂 本地选择视频文件
2. 🖱️ 在视频首帧上画激光线
3. 🔄 **SFTP 自动上传视频**到服务器 `data/` 目录
4. 📋 **SFTP 自动同步配置** `config.json` 到服务器
5. 🔥 **SSH 远程触发** `traffic_monitor.py` 后台运行
6. 📝 服务器推理日志写入 `detection.log`

### 6.5 手动远程运行

如果不使用 `setup_roi.py`，也可以手动操作：

```bash
# 1. 上传视频到服务器
scp -P 10743 your_video.mp4 root@your-server:/root/autodl-tmp/FlowPlate-Analyzer/data/

# 2. SSH 登录服务器
ssh -p 10743 root@your-server

# 3. 修改 config.json 中的 video_path 和激光线坐标

# 4. 后台运行（断开 SSH 不中断）
cd /root/autodl-tmp/FlowPlate-Analyzer
nohup python traffic_monitor.py > detection.log 2>&1 &

# 5. 查看日志
tail -f detection.log
```

### 6.6 Docker 容器部署（可选）

创建 `Dockerfile`：

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    fonts-wqy-zenhei \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

ENV QT_QPA_PLATFORM=offscreen

CMD ["python", "traffic_monitor.py"]
```

构建与运行：

```bash
# 构建镜像（需提前将 best.onnx, yolov8m.pt, 视频文件放入项目目录）
docker build -t traffic-monitor .

# 运行容器
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app traffic-monitor
```

> 输出文件 `yolo_hy_plate_result.mp4` 和 `recognition_results.csv` 将保存在宿主机的 `./output/` 目录。

---

## 7. 常见问题排查

### 7.1 中文字符显示为 `??` 或方块

**原因**: Linux 系统缺少中文字体  
**解决**:
```bash
apt-get update && apt-get install -y fonts-wqy-zenhei
```
程序启动时会自动探测系统中可用的中文字体（按优先级依次尝试 Wqy-Zenhei → Noto CJK → Windows 系统字体），无需手动修改代码中的 `FONT_PATH`。若自动探测失败，终端会打印字体安装提示。

### 7.2 CUDA Out of Memory

**原因**: 显存不足  
**解决**:
- 降低 `SCALE` 值（如 0.3）
- 在 `traffic_monitor.py` 中设置 `device='cpu'`（牺牲速度保功能）

### 7.3 车牌识别率低

**可能原因与排查步骤**：

| 原因 | 排查方法 | 解决方案 |
|---|---|---|
| 车牌检测模型未收敛 | 查看 `runs/flow_plate_m/results.png` | 增加 epochs 或调整学习率 |
| 激光线位置不佳 | 检查 `debug_capture/` 中抓拍的车身裁剪图 | 用 `setup_roi.py` 重新划线 |
| 视频分辨率过低 | 检查原始视频分辨率 | 确保输入视频 ≥ 1080p |
| 极端角度/遮挡 | 查看 FAILED 案例 | 补充训练数据覆盖 Corner Cases |

### 7.4 训练数据格式错误

**症状**: 训练时 `labels` 全为 0 或报错  
**排查**:
```bash
# 检查标签文件是否正确生成
ls ./data/train/*.txt | head -5
cat ./data/train/XXXXX.txt  # 应看到类似 "0 0.5234 0.6123 0.0891 0.0342"
```

### 7.5 虚拟激光线触发不准

**调试方法**:
- 增大 `max_dist`（默认 15px）降低触发门槛
- 在 `traffic_monitor.py` 中输出 `cx`, `cy` 观察车辆中心点轨迹
- 确保激光线横跨车道，不要平行于车流方向

### 7.6 OpenCV 窗口无法关闭 / 程序卡死

```bash
# 强制杀死所有 Python 进程
# Windows:
taskkill /F /IM python.exe

# Linux:
pkill -9 python
```

### 7.7 GPU 不可用 / CUDA 错误

**症状**: 推理极慢（< 5 FPS）或报 `CUDA not available`  
**排查步骤**:

```bash
# 1. 检查驱动
nvidia-smi

# 2. 检查 PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 3. 检查 ONNX Runtime 是否识别 GPU
python -c "import onnxruntime; print(onnxruntime.get_device())"
```

**常见解决方案**:

| 症状 | 方案 |
|---|---|
| `nvidia-smi` 正常但 `torch.cuda.is_available()` 为 False | PyTorch 版本与 CUDA 不匹配，重新安装对应版本 |
| ONNX Runtime 使用 CPU | 安装 `onnxruntime-gpu` 替代 `onnxruntime` |
| 显存不足 | 降低 `SCALE` 至 0.3，或改用 CPU 推理 |

### 7.8 远程部署连接失败

**症状**: `setup_roi.py` 无法连接远程服务器  
**排查**:
- 确认 AutoDL 实例已开机且 SSH 端口正确（AutoDL 控制台可查）
- 检查本地防火墙是否阻止出站连接
- 尝试手动 SSH 登录验证凭证：`ssh -p <PORT> root@<IP>`

### 7.9 训练后项目目录清理

训练过程中产生的冗余文件建议清理以保持仓库整洁：

```bash
# 删除 Jupyter Notebook 检查点缓存（已在 .gitignore 中）
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

# 如果训练输出路径有嵌套重复（如 runs/detect/runs/detect/），可清理后重新训练
# 新版本 train.py 已将 project 改为 "runs"，不会产生嵌套问题
```

---

## 8. Linux 生产环境部署（systemd 服务）

将 `traffic_monitor.py` 注册为系统服务，实现开机自启动和崩溃自动重启。

### 8.1 创建服务文件

```bash
sudo nano /etc/systemd/system/traffic-monitor.service
```

填入以下内容（请替换路径为实际路径）：

```ini
[Unit]
Description=Smart Traffic Monitor - Plate Recognition
After=network.target

[Service]
Type=simple
WorkingDirectory=/root/autodl-tmp/FlowPlate-Analyzer
ExecStart=/usr/bin/python3 /root/autodl-tmp/FlowPlate-Analyzer/traffic_monitor.py
Restart=on-failure
RestartSec=10
StandardOutput=append:/root/autodl-tmp/FlowPlate-Analyzer/detection.log
StandardError=append:/root/autodl-tmp/FlowPlate-Analyzer/detection.log
Environment="QT_QPA_PLATFORM=offscreen"

[Install]
WantedBy=multi-user.target
```

### 8.2 启用与管理

```bash
# 重载配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start traffic-monitor

# 设置开机自启
sudo systemctl enable traffic-monitor

# 查看运行状态
sudo systemctl status traffic-monitor

# 查看日志
journalctl -u traffic-monitor -f
```

---

## 📊 附录：完整工作流速查

```text
┌─────────────────────────────────────────────────────────────────┐
│                     从零到部署完整流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 环境搭建                                                     │
│     pip install -r requirements.txt                             │
│     apt-get install fonts-wqy-zenhei  (Linux only)              │
│                                                                 │
│  2. 数据集准备                                                   │
│     ├── 下载 CCPD 数据集                                         │
│     ├── python split_dataset.py    (8:1:1 划分)                  │
│     └── python ccpd_to_yolo.py     (格式转换)                     │
│                                                                 │
│  3. 模型训练                                                     │
│     └── python train.py             (训练 ~100 epochs)           │
│                                                                 │
│  4. 部署推理                                                     │
│     ├── [本地-直接] python traffic_monitor.py                     │
│     ├── [本地-可视] python setup_roi_local.py → 画线 → 自动运行    │
│     └── [远程部署] python setup_roi.py → 画线 → 自动上传+运行      │
│                                                                 │
│  5. 结果输出                                                     │
│     ├── yolo_hy_plate_result.mp4   (渲染视频)                    │
│     ├── recognition_results.csv    (统计报表)                    │
│     └── debug_capture/             (抓拍图)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

> ⚠️ **授权声明**: 本项目代码仅供学习与科研参考，禁止用于任何非法用途。训练所用 CCPD 数据集版权归原作者所有。
