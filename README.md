# 视网膜血管分割演示系统（Flask + PyTorch）

一个用于**眼底图像视网膜血管分割**的本地演示系统：浏览器上传图像，后端进行推理并返回**分割掩膜**与**叠加可视化**。

---

## 功能特性

- **Web 演示**：访问首页，上传图像即可看到分割结果（原图 / 掩膜 / 叠加）。
- **多图上传**：一次可上传多张图片（默认最多 20 张）。
- **格式兼容**：支持 `JPG / PNG / TIFF / PPM / PGM / PBM`（含部分 PPM/PGM 的兼容读取逻辑）。
- **稳健推理**：每张图独立处理，失败不影响同批其它图片，返回 `errors` 便于定位问题。
- **训练曲线生成**：可生成不同训练版本对比图（见 `plot_all_history.py`）。

---

## 模型说明（简要）

当前推理模型在 `app.py` 中定义为 `ImprovedUNet`，核心结构为：

- U-Net 主干（编码器/解码器 + skip connection）
- **ResidualBlock**（残差卷积块）
- **ChannelAttention**（通道注意力）
- 末端 **Multi-Scale Fusion**（3x3 与 5x5 多尺度特征融合）
- 输出为 1 通道 sigmoid mask（阈值 0.5）

> 说明：页面上显示的 Dice/敏感度/特异性来自 **checkpoint 中保存的指标**（或默认值），并非对你上传的图片实时计算（实时计算需要 GT 掩膜）。

---

## 目录结构（关键文件）

```
demo_system/
  app.py                       # Flask 推理服务（入口）
  templates/
    index.html                 # 前端页面
  checkpoints/
    best_model_800.pth          # 优先加载（如存在）
    best_model.pth
    final_model.pth
    history.json                # 训练日志（v1）
    history_v2.json             # 训练日志（v2）
    history_v3.json             # 训练日志（v3）
  plot_all_history.py           # 训练曲线对比图生成
  uploads/                      # 上传的临时文件夹（运行时自动创建/清理）
```

---

## 环境依赖

建议使用 Python 3.9+（Windows 可用）。

主要依赖（按 `app.py` 导入推断）：

- `flask`
- `torch`
- `opencv-python`
- `numpy`
- `Pillow`

如需绘图：

- `matplotlib`

---

## 快速开始（本地运行）

在 `demo_system/` 目录下运行：

```bash
python app.py
```

启动后访问：

- `http://127.0.0.1:5000`

---

## 模型权重加载逻辑

服务启动时会按优先级尝试加载：

1. `checkpoints/best_model_800.pth`
2. `checkpoints/best_model.pth`
3. `checkpoints/final_model.pth`

加载成功后会设置：

- `LOADED_CHECKPOINT`：当前实际加载的 checkpoint 路径
- `MODEL_DICE / MODEL_SENS / MODEL_SPEC`：从 checkpoint 读取（若 checkpoint 内包含这些字段）

推理接口也会回传当前模型信息：

- 返回字段 `model_info.checkpoint` 可用于确认正在服务的权重文件。

---

## 接口说明

### `GET /`

返回上传与展示页面（`templates/index.html`）。

### `POST /predict`

支持表单字段：

- `files`：多文件上传（推荐）
- `file`：兼容单文件字段（前端旧版本也可用）

返回（成功）：

- `success: true`
- `results: []`：每个元素包含：
  - `filename`
  - `original`：base64 PNG
  - `mask`：base64 PNG
  - `overlay`：base64 PNG
  - `model_info.checkpoint`
  - `metrics`（来自 checkpoint/默认值）
- `errors: []`：同批次失败项（如有）

---

## 生成训练曲线对比图

在 `demo_system/` 目录运行：

```bash
python plot_all_history.py
```

输出文件：

- `demo_system/all_training_curves.png`

中文标题支持：

- 脚本已设置 `Microsoft YaHei / SimHei` 等字体作为优先候选。

如需弹窗显示（默认不弹窗，避免卡住）：

```bash
set SHOW_PLOT=1
python plot_all_history.py
```

---

## 常见问题（Troubleshooting）

### 1) 页面显示的 Dice 没变化

- 先确认服务端日志里“已加载模型”的实际路径
- 或在 `/predict` 返回里看 `model_info.checkpoint`
- 然后**重启**服务（不要只刷新页面）

### 2) Windows 控制台乱码 / 编码问题

- 若你的终端编码为 GBK，请避免在 `print()` 里输出某些 Unicode 符号（例如 ✅）。
- 本项目已移除容易触发编码异常的输出符号，确保不会因为打印而导致权重加载被异常中断。

### 3) 上传多张大图报 413

- `MAX_CONTENT_LENGTH` 已设置为 80MB；如仍超出，可在 `app.py` 增大。

---

## 演示建议（答辩友好）

- 演示时先上传 1 张展示流程，再多选 3–5 张展示“批量推理 + 部分失败不影响整体”的鲁棒性。
- 强烈建议答辩前用 `model_info.checkpoint` 确认正在运行的是你新训练的权重文件。

