# 视网膜血管分割演示系统（FastAPI + PyTorch）

在线演示服务：上传眼底图像，返回分割掩膜、叠加可视化和模型指标。

---

## 1. 运行方式

### 本地开发

```bash
cd demo_system
cp .env.example .env
pip install -r requirements.txt
python run.py
```

访问：`http://127.0.0.1:5000/`

### 生产启动

```bash
gunicorn -k uvicorn.workers.UvicornWorker app.main:app -w 2 -b 0.0.0.0:5000 --timeout 120
```

### Docker（CPU）

```bash
docker compose up --build -d
```

### Docker（GPU）

```bash
docker compose --profile gpu up --build -d
```

GPU 容器默认映射到端口 5001，需要 NVIDIA Container Toolkit。

---

## 2. 配置（.env）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_UPLOADS` | 20 | 单次最多图片数 |
| `THRESHOLD` | 0.5 | 分割阈值 |
| `UPLOAD_DIR` | ./uploads | 上传临时目录 |
| `LOG_LEVEL` | INFO | 日志级别 |
| `RATE_LIMIT_RPM` | 60 | 每 IP 每分钟最大请求数 |

---

## 3. API

### `GET /`
返回上传与结果展示页面。

### `POST /predict`
表单字段：`files`（多文件）或 `file`（兼容旧调用）。

返回结构（简化）：
```json
{
  "success": true,
  "results": [
    {
      "filename": "example.png",
      "original": "<base64>",
      "mask": "<base64>",
      "overlay": "<base64>",
      "metrics": {"dice": 86.5, "sensitivity": 82.0, "specificity": 97.5},
      "model_info": {"checkpoint": "checkpoints/best_model_800.pth"}
    }
  ],
  "errors": []
}
```

### `GET /health`
健康检查，返回 `{"success": true, "status": "ok"}`。

### `GET /model_info`
当前 checkpoint 路径、指标、加载时间。

### `POST /evaluate`
上传 `image` + `mask`（GT），返回在线评估指标。

---

## 4. 目录结构

```text
demo_system/
├── app/
│   ├── main.py                      # FastAPI 入口、中间件
│   ├── api/routes_predict.py        # 路由
│   ├── core/
│   │   ├── config.py                # 配置
│   │   ├── errors.py                # 统一异常
│   │   ├── logging.py               # 日志 + request_id
│   │   └── schemas.py               # 类型定义
│   ├── inference/
│   │   ├── model_def.py             # ImprovedUNet
│   │   ├── model_loader.py          # checkpoint 加载
│   │   ├── preprocess.py            # 预处理
│   │   ├── postprocess.py           # 后处理 + 评估指标
│   │   └── service.py               # 推理服务
│   └── utils/image_codec.py         # base64 编码
├── tests/
│   ├── test_postprocess.py
│   └── test_service.py
├── templates/index.html
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── run.py
```

---

## 5. 工程能力清单

- [x] 模块化拆分（路由/推理/配置/日志分离）
- [x] 统一异常处理（`InputError`/`ModelError`/`ProcessingError`）
- [x] 统一错误响应（含 `request_id` + `code`）
- [x] 结构化日志（时间/级别/request_id/耗时）
- [x] 请求限流（基于 IP 的 RPM 限制）
- [x] 上传体积限制中间件
- [x] 健康检查 + 模型信息接口
- [x] 在线评估接口（上传 GT mask 算指标）
- [x] 单元测试（pytest）
- [x] Docker + docker-compose（CPU/GPU）
- [x] CI（GitHub Actions: lint + test）
- [x] `.env` 配置外置

---

## 6. 常见问题

### 推理结果全黑
检查启动日志中 `model_info.checkpoint` 是否为 `(not loaded)`。将 `.pth` 放入 `checkpoints/` 并重启。

### 429 Too Many Requests
调大 `.env` 中 `RATE_LIMIT_RPM`，或在反向代理层做限流。

### GPU 部署
1. 安装 NVIDIA Container Toolkit
2. `docker compose --profile gpu up --build -d`
3. 确保 Dockerfile 基础镜像换为 `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`（按需修改）
