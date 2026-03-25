# 视网膜血管分割演示系统（demo_system）

## 目标
- 保持现有业务行为（`/predict` 返回结构和前端展示兼容）。
- 在此基础上补齐工程化能力：配置、日志、测试、健康检查、模型信息、容器部署、实验闭环入口。

## 运行

### 1) 本地开发

```bash
cd demo_system
cp .env.example .env
pip install -r requirements.txt
python run.py
```

访问：`http://127.0.0.1:5000/`

### 2) 生产启动（示例）

```bash
gunicorn -k uvicorn.workers.UvicornWorker app.main:app -w 2 -b 0.0.0.0:5000 --timeout 120
```

### 3) Docker

```bash
docker compose up --build -d
```

## 配置（.env）

- `MAX_UPLOADS`：单次最多图片数（默认 20）
- `THRESHOLD`：分割阈值（默认 0.5）
- `UPLOAD_DIR`：上传临时目录（默认 `./uploads`）
- `LOG_LEVEL`：日志级别（默认 `INFO`）

## API

### `GET /`
返回页面。

### `POST /predict`
兼容旧结构（单图时顶层保留 `original/mask/overlay/metrics`）。

### `GET /health`
健康检查。

### `GET /model_info`
返回当前 checkpoint、checkpoint 指标、模型加载时间。

### `POST /evaluate`
上传 `image` + `mask`，输出在线评估指标（dice/sensitivity/specificity），用于评估闭环。

## 目录

```text
demo_system/
├── app/
│   ├── api/routes_predict.py
│   ├── core/{config.py,errors.py,logging.py,schemas.py}
│   ├── inference/{model_def.py,model_loader.py,preprocess.py,postprocess.py,service.py}
│   ├── utils/image_codec.py
│   └── main.py
├── tests/
│   ├── test_postprocess.py
│   └── test_service.py
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── run.py
```

## 实验体系入口

仓库根目录新增：
- `experiments/configs/*.yaml`
- `experiments/ablation_plan.md`

可与现有训练脚本联动，训练后统一输出 `metrics.json` 并汇总到实验报告。
