# 视网膜血管分割演示系统（Flask + PyTorch）

一个用于**眼底图像视网膜血管分割**的本地演示系统：浏览器上传图像，后端进行推理并返回**分割掩膜**与**叠加可视化**。

---

## 功能特性

- **Web 演示**：访问首页，上传图像即可看到分割结果（原图 / 掩膜 / 叠加）。
- **多图上传**：一次可上传多张图片（默认最多 20 张）。
- **格式兼容**：支持 `JPG / PNG / TIFF / PPM / PGM / PBM`（含部分 PPM/PGM 的兼容读取逻辑）。
- **稳健推理**：每张图独立处理，失败不影响同批其它图片，返回 `errors` 便于定位问题。
- **模块化结构**：按模型定义、预处理、模型加载、服务、路由、配置拆分，便于继续维护。
- **训练曲线生成**：可生成不同训练版本对比图（见 `plot_all_history.py`）。

---

## 当前后端结构

```text
.
├── app.py                  # 兼容旧入口，转发到 app.main
├── run.py                  # 新的推荐启动入口
├── app/
│   ├── __init__.py
│   ├── main.py             # Flask app 组装
│   ├── model_def.py        # 模型定义
│   ├── preprocess.py       # 预处理
│   ├── image_codec.py      # 图像读取 / base64 编码
│   ├── model_loader.py     # checkpoint 加载与指标管理
│   ├── service.py          # 单图 / 批处理推理逻辑
│   ├── routes_predict.py   # 预测路由
│   └── config.py           # 配置项
├── templates/
│   └── index.html
└── checkpoints/
    ├── best_model_800.pth
    ├── best_model.pth
    └── final_model.pth
```

---

## 模型说明（简要）

当前推理模型在 `app/model_def.py` 中定义为 `ImprovedUNet`，核心结构为：

- U-Net 主干（编码器/解码器 + skip connection）
- **ResidualBlock**（残差卷积块）
- **ChannelAttention**（通道注意力）
- 末端 **Multi-Scale Fusion**（3x3 与 5x5 多尺度特征融合）
- 输出为 1 通道 sigmoid mask（默认阈值 0.5）

> 说明：页面上显示的 Dice/敏感度/特异性来自 **checkpoint 中保存的指标**（或默认值），并非对你上传的图片实时计算（实时计算需要 GT 掩膜）。

---

## 环境依赖

建议使用 Python 3.9+。

主要依赖：

- `flask`
- `torch`
- `opencv-python`
- `numpy`
- `Pillow`

如需绘图：

- `matplotlib`

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 快速开始（推荐）

### 方式 1：使用新的启动入口

```bash
python run.py
```

### 方式 2：继续使用旧入口

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

- `loaded_checkpoint`：当前实际加载的 checkpoint 路径
- `dice / sensitivity / specificity`：从 checkpoint 读取（若 checkpoint 内包含这些字段）

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

## 全链路回归建议

```bash
python -m compileall app app.py run.py
python run.py
```

如果本地已安装 Flask / PyTorch / OpenCV，可再手动上传 1 张测试图确认：

- 首页可打开
- `/predict` 可返回 `original / mask / overlay`
- 页面显示的模型 checkpoint 与指标正常

---

## 常见问题（Troubleshooting）

### 1) 页面显示的 Dice 没变化

- 先确认服务端日志里“已加载模型”的实际路径
- 或在 `/predict` 返回里看 `model_info.checkpoint`
- 然后**重启**服务（不要只刷新页面）

### 2) 上传多张大图报 413

- `MAX_CONTENT_LENGTH` 已放入 `app/config.py`；如仍超出，可调大对应配置。

### 3) 想调整魔法数字

以下参数已迁移到 `app/config.py`：

- 上传目录
- 最大上传体积
- 单次最大文件数
- 默认指标
- 输入尺寸
- CLAHE 参数
- Gamma
- 阈值
- host / port / debug

---

## 迁移说明

本次重构遵循“先复制、再抽模块、最后补入口与文档”的顺序，尽量保持原有推理逻辑不变，只做职责拆分与配置收口，方便后续继续演进。
