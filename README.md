# 视网膜血管分割 — Improved U-Net

基于 PyTorch 的**眼底图像血管分割**项目：改进 U-Net、配置化训练与消融、测试集逐样本评估、FastAPI 在线演示与 Docker 部署。

**作者：** 李国敬 · **许可证：** MIT

---

## 一、项目背景

糖尿病视网膜病变等疾病的筛查依赖眼底照相。血管结构是重要生物标志，**自动分割视网膜血管**可辅助病灶检测与病程评估。传统 U-Net 在细血管与低对比度区域易出现断裂或漏检。本项目在经典编码器–解码器框架上引入**残差块、通道注意力与多尺度融合**，并结合**绿色通道 + CLAHE + 伽马校正**等预处理，在保持参数量适中的前提下提升分割连续性与边界精度。仓库同时提供**消融实验配置**、**逐样本指标**与**演示服务**，便于复现与展示。

---

## 二、主要创新点（相对基准 U-Net）

| 方向 | 内容 |
|------|------|
| 网络结构 | **残差卷积块**稳定深层训练；**通道注意力(Channel Attention)**突出血管相关特征；解码端 **3×3 + 5×5 多尺度分支**再融合，兼顾细线与粗结构。 |
| 预处理 | 使用**绿色通道**；**CLAHE** 增强局部对比度；**伽马校正**抑制过曝/欠曝，与训练管线一致。 |
| 训练目标 | 支持 **Combo（加权 BCE + Dice + 假阴性惩罚）**、**BCE+Dice** 等（见 `run_experiment.py` 与 `experiments/configs/*.yaml`）。 |
| 工程与实验 | **YAML 驱动** `run_experiment.py`（训练 + 测试集评估 + `metrics.json`）；**逐样本** `per_sample_metrics.csv`；**云同步目录**下权重与指标双写；**FastAPI** 演示与 **CI**。 |

---

## 三、文件夹结构（详细）

```text
retinal-vessel-segmentation-unet/
├── README.md                      # 本说明
├── run_experiment.py              # 一键：读 YAML → 训练 → 测试集评估 → metrics.json（含 per_sample）
├── evaluate.py                    # 仅评估（支持多路径解析 checkpoint、同步写指标）
├── report.py                      # 扫描 experiments/results/*/metrics.json → summary.csv / summary.md
├── train_final.py                 # 早期固定脚本训练（800 张等场景，可作参考）
├── train.py / train_v2.py / train_v3.py   # 历史训练变体（DRIVE / 不同损失等）
├── app.py                         # 历史单文件 Flask 演示（可选，主入口以 demo_system 为准）
├── merge_datasets.py              # 数据集整理工具
├── verify_data.py                 # 数据统计
├── plot_history.py / plot_all_history.py  # 训练曲线可视化
│
├── experiments/
│   ├── ablation_plan.md           # 8 组消融说明
│   ├── configs/                   # 每组实验一个 YAML（模型开关、预处理、训练超参、阈值）
│   │   ├── baseline.yaml
│   │   ├── no_attention.yaml
│   │   ├── no_residual.yaml
│   │   ├── no_multiscale.yaml
│   │   ├── no_clahe.yaml
│   │   ├── no_gamma.yaml
│   │   ├── input_256.yaml
│   │   └── loss_bce_dice.yaml
│   ├── results/                   # 本机默认实验输出根目录
│   │   ├── summary.csv            # report.py 汇总（多实验对比表）
│   │   ├── summary.md
│   │   ├── summary_visualization.png   # python experiments/plot_summary_results.py
│   │   └── <实验名>/            # 如 loss_bce_dice/
│   │       ├── best_model.pth
│   │       ├── history.json      # 每 epoch 曲线
│   │       ├── metrics.json      # 平均指标 + per_sample 列表
│   │       ├── per_sample_metrics.csv
│   │       ├── per_sample_sorted_by_dice.csv
│   │       ├── per_sample_summary.md
│   │       └── per_sample_visualization.png
│   ├── summarize_per_sample_csv.py
│   ├── plot_per_sample_sorted_csv.py
│   └── plot_summary_results.py
│
├── results/                       # 可选：从服务器同步的结果镜像（与 experiments/results 二选一或并存）
│   ├── experiments/results/<实验名>/…   # 与上同结构
│   ├── plot_ablation_curves.py    # 多实验 val_dice 曲线对比
│   └── ablation_val_dice_curves.png
│
├── dataset/                       # 数据根目录（通常不提交 Git，见 .gitignore）
│   ├── training/
│   │   ├── images/                # 训练图像，如 *.png
│   │   └── mask/                  # 与 images 同文件名的二值标注
│   └── test/
│       ├── images/
│       └── mask/
│
├── checkpoints/                   # 根目录权重（可选，与 demo / 训练脚本约定一致）
│
├── demo_system/                   # FastAPI 在线演示
│   ├── app/                       # main、api、core、inference、utils
│   ├── models/best_model.pth      # 演示优先加载（可放当前最优实验权重）
│   ├── templates/index.html
│   ├── requirements.txt
│   ├── run.py
│   ├── Dockerfile / docker-compose.yml
│   ├── .env.example
│   └── README.md                  # 端口、API、Docker、限流等细节
│
└── .github/workflows/ci.yml       # push/PR：lint + pytest（工作目录 demo_system）
```

---

## 四、数据集说明

- **放置位置：** 仓库根下 `dataset/`（若路径不同，在 YAML 中修改 `dataset_dir`）。
- **目录约定：**
  - `dataset/training/images` + `dataset/training/mask`
  - `dataset/test/images` + `dataset/test/mask`
- **文件规则：** 图像与掩膜**同名**（如 `kaggle_001.png`）；当前脚本以 **PNG** 为主。
- **掩膜：** 单通道灰度，血管区域通常 >128 视为前景（与 `FundusDataset` / 评估逻辑一致）。

数据体量大时请勿提交 Git；本地准备好后可直接训练/评估。

---

## 五、环境安装与运行步骤

### 5.1 建议环境

- **Python 3.11**（3.10+ 一般可用；避免仅 3.14 且无预编译 wheel 的组合）
- **PyTorch**：按 [官网说明](https://pytorch.org/) 安装 CPU 或 CUDA 版本

### 5.2 训练与实验（仓库根目录）

```bash
# 核心依赖（训练 + 实验）
pip install torch numpy opencv-python Pillow pyyaml tqdm

# 单个实验：训练 + 测试集评估 + 写 metrics.json（含逐样本）
python run_experiment.py --config experiments/configs/baseline.yaml

# 仅评估（自动尝试 experiments/results 与 results/experiments/results 下的权重）
python run_experiment.py --config experiments/configs/loss_bce_dice.yaml --eval-only

# 或使用独立评估脚本
python evaluate.py --config experiments/configs/loss_bce_dice.yaml

# 汇总所有子实验 metrics → summary.csv / summary.md
python report.py

# 逐样本 CSV → 摘要 Markdown / 排序表 / 统计 JSON
python experiments/summarize_per_sample_csv.py \
  --input experiments/results/loss_bce_dice/per_sample_metrics.csv

# 逐样本排序 CSV → 四宫格可视化 PNG（--input 须为真实路径，勿写 ...）
python experiments/plot_per_sample_sorted_csv.py \
  --input experiments/results/loss_bce_dice/per_sample_sorted_by_dice.csv

# summary.csv → 汇总柱状图 + 表格 PNG
python experiments/plot_summary_results.py
```

YAML 中 `training.device` 改为 `cuda` 可在 GPU 上训练（需本机 CUDA 可用）。

### 5.3 在线演示（demo_system）

```bash
cd demo_system
cp .env.example .env   # 按需修改
pip install -r requirements.txt
python run.py
# 浏览器访问 http://127.0.0.1:5000
```

生产可用 Gunicorn + Uvicorn Worker，或 `docker compose`；详见 `demo_system/README.md`。

### 5.4 端口占用

若 `5000` 已被占用，可改用例如：

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 5001
```

---

## 六、实验结果说明

- **权威汇总：** 在跑完各子实验并生成 `experiments/results/<name>/metrics.json` 后，执行 **`python report.py`**，查看：
  - `experiments/results/summary.csv`
  - `experiments/results/summary.md`
- **指标含义：** **Dice / IoU** 反映重叠度；**Sensitivity** 越高漏检血管越少；**Specificity** 越高背景误判越少。`metrics.json` 中的 **per_sample** 为测试集**每张图**的上述指标 + 推理耗时；`report.py` 汇总行为**各实验在整测试集上的平均**（来自各 `metrics.json` 顶层字段）。
- **可视化：**
  - 多实验训练曲线：`results/plot_ablation_curves.py`（默认读 `results/experiments/results/*/history.json`）
  - 单实验逐样本：`experiments/plot_per_sample_sorted_csv.py`
  - 汇总表图：`experiments/plot_summary_results.py`

> **注意：** 仓库内随附的 `summary.csv` 可能仅为示例或部分实验；**完整对比请以您本机运行 `report.py` 后的表格为准**。在完整消融中，常见现象包括：去掉注意力或多尺度后 Dice 略降；**BCE+Dice（无强假阴性项）** 在部分数据划分上可能略优于 Combo，需以您复现实验为准。

---

## 七、消融矩阵（8 组）

| 配置 | 说明 |
|------|------|
| `baseline.yaml` | 残差 + 通道注意力 + 多尺度，默认预处理与输入 512 |
| `no_attention.yaml` | 关闭通道注意力 |
| `no_residual.yaml` | 残差改为普通卷积块 |
| `no_multiscale.yaml` | 关闭多尺度头 |
| `no_clahe.yaml` / `no_gamma.yaml` | 分别关闭 CLAHE / 伽马校正 |
| `input_256.yaml` | 输入 256×256 |
| `loss_bce_dice.yaml` | 损失为 BCE + Dice（与 Combo 对比） |

详见 `experiments/ablation_plan.md`。

---

## 八、引用与扩展

若本项目对您的论文或竞赛有帮助，可注明仓库与作者。扩展方向包括：多数据集泛化、后处理（CRF/细化）、Transformer 编码器、半监督与域适应等。

---

## 九、相关文档

- `demo_system/README.md` — API、环境变量、Docker/GPU、限流与排障  
- `experiments/ablation_plan.md` — 消融执行命令与指标说明  

如有问题，欢迎提 Issue 或 PR。
