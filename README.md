# 视网膜血管分割演示系统

本仓库保留训练代码，同时将在线演示服务重构为 `demo_system/` 目录下的模块化 FastAPI 应用，便于维护模型定义、加载、预处理、后处理与接口层。

## 运行方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Demo

```bash
cd demo_system
python run.py
```

启动后访问：

- `http://127.0.0.1:8000/`
- Swagger 文档：`http://127.0.0.1:8000/docs`

## 接口说明

### `GET /`

返回演示页面，支持多图上传与结果展示。

### `POST /predict`

- 表单字段：`files`（推荐，多文件）或 `file`（兼容旧调用）
- 支持格式：`PNG / JPG / JPEG / TIFF / PPM / PGM / PBM`
- 默认最多 20 张图，总上传体积默认限制 80MB

返回 JSON 结构：

```json
{
  "success": true,
  "results": [
    {
      "filename": "example.png",
      "original": "<base64 PNG>",
      "mask": "<base64 PNG>",
      "overlay": "<base64 PNG>",
      "metrics": {
        "dice": 86.5,
        "sensitivity": 82.0,
        "specificity": 97.5
      },
      "model_info": {
        "checkpoint": "checkpoints/best_model_800.pth"
      }
    }
  ],
  "errors": []
}
```

## 目录说明

```text
demo_system/
├── app/
│   ├── api/
│   │   └── routes_predict.py      # /predict 和 / 路由
│   ├── core/
│   │   ├── config.py              # 上传限制、支持格式、阈值、路径
│   │   └── schemas.py             # 响应结构
│   ├── inference/
│   │   ├── model_def.py           # ResidualBlock、ChannelAttention、ImprovedUNet
│   │   ├── model_loader.py        # checkpoint 候选加载与指标读取
│   │   ├── postprocess.py         # 阈值、resize、overlay
│   │   ├── preprocess.py          # robust_imread、preprocess
│   │   └── service.py             # process_one/process_batch 主流程
│   ├── utils/
│   │   └── image_codec.py         # base64 编码等通用转换
│   └── main.py                    # 应用入口、路由注册、启动配置
├── run.py                         # 本地启动入口
└── templates/
    └── index.html                 # 演示页面
```

## 模型加载逻辑

服务启动时按以下顺序尝试加载权重：

1. `checkpoints/best_model_800.pth`
2. `checkpoints/best_model.pth`
3. `checkpoints/final_model.pth`

如果 checkpoint 中包含 `dice`、`sensitivity`、`specificity`，系统会自动读取并在接口返回中携带；否则回退到默认演示指标。

## 说明

- 训练脚本、历史曲线与 checkpoint 仍保留在仓库根目录原位置。
- 原来的单文件 `app.py` 未参与新的 `demo_system` 启动流程；新的入口是 `demo_system/run.py`。