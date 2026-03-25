# Retinal Vessel Segmentation — Improved U-Net

基于 PyTorch 的视网膜血管分割系统，包含训练、评估、消融实验和在线演示。

## 项目结构

```text
retinal-vessel-segmentation-unet/
├── run_experiment.py          # 配置驱动的一键训练+评估入口
├── evaluate.py                # 独立评估脚本
├── report.py                  # 汇总所有实验到 CSV/Markdown
├── train_final.py             # 原始训练脚本（参考保留）
├── experiments/
│   ├── configs/               # 实验配置（每个 YAML 一组实验）
│   │   ├── baseline.yaml
│   │   ├── no_attention.yaml
│   │   ├── no_residual.yaml
│   │   ├── no_multiscale.yaml
│   │   ├── no_clahe.yaml
│   │   ├── no_gamma.yaml
│   │   ├── input_256.yaml
│   │   └── loss_bce_dice.yaml
│   ├── ablation_plan.md       # 消融计划说明
│   └── results/               # 实验产出（metrics.json + summary）
├── checkpoints/               # 模型权重
├── dataset/                   # 数据集（training/test）
├── demo_system/               # 在线演示服务（FastAPI）
│   ├── app/                   # 服务端代码
│   ├── templates/             # 前端页面
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── README.md              # Demo 专属文档
└── .github/workflows/ci.yml   # CI 自动化
```

## 快速开始

### 1. 环境准备

```bash
# 推荐 Python 3.11
pip install torch numpy opencv-python Pillow pyyaml tqdm
```

### 2. 运行实验

```bash
# 单个实验（训练 + 评估 + 写 metrics.json）
python run_experiment.py --config experiments/configs/baseline.yaml

# 换种子重跑
python run_experiment.py --config experiments/configs/baseline.yaml --seed 3407

# 仅评估已有 checkpoint
python run_experiment.py --config experiments/configs/baseline.yaml --eval-only
```

### 3. 汇总结果

```bash
python report.py
# 输出: experiments/results/summary.csv + summary.md
```

### 4. 启动 Demo 服务

```bash
cd demo_system
pip install -r requirements.txt
python run.py
# 访问 http://127.0.0.1:5000
```

## 消融实验矩阵（8 组）

| # | 实验 | 变更 |
|---|------|------|
| 1 | baseline | 全模块（Residual + CA + Multi-scale） |
| 2 | no_attention | 去掉 Channel Attention |
| 3 | no_residual | 去掉残差连接 |
| 4 | no_multiscale | 去掉多尺度头 |
| 5 | no_clahe | 去掉 CLAHE 预处理 |
| 6 | no_gamma | 去掉 Gamma 校正 |
| 7 | input_256 | 输入 256x256 |
| 8 | loss_bce_dice | 损失改为 BCE+Dice |

详见 `experiments/ablation_plan.md`。

## 评估指标

- **Dice**：分割主指标
- **IoU**：交并比
- **Sensitivity**：血管检出率
- **Specificity**：背景正确率
- **Params**：模型参数量
- **Inference(ms)**：单张推理时延

## 部署

### CPU

```bash
cd demo_system
docker compose up --build -d
```

### GPU（需要 NVIDIA Docker）

```bash
cd demo_system
docker compose --profile gpu up --build -d
```

详见 `demo_system/README.md`。

## CI

推送到 `main` 分支或创建 PR 时，GitHub Actions 自动运行 lint + 单元测试。

## 作者

李国敬
