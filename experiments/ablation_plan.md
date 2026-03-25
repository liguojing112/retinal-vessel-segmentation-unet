# 消融实验计划（8 组）

## 实验矩阵

| # | 实验名 | 配置文件 | 变更内容 |
|---|--------|----------|----------|
| 1 | baseline | `configs/baseline.yaml` | 全模块开启（Residual + CA + Multi-scale） |
| 2 | no_attention | `configs/no_attention.yaml` | 去掉 Channel Attention |
| 3 | no_residual | `configs/no_residual.yaml` | 去掉残差连接（改用 PlainConvBlock） |
| 4 | no_multiscale | `configs/no_multiscale.yaml` | 去掉多尺度头（改用单 3x3 conv） |
| 5 | no_clahe | `configs/no_clahe.yaml` | 预处理去掉 CLAHE |
| 6 | no_gamma | `configs/no_gamma.yaml` | 预处理去掉 Gamma 校正 |
| 7 | input_256 | `configs/input_256.yaml` | 输入尺寸 256x256（vs baseline 512） |
| 8 | loss_bce_dice | `configs/loss_bce_dice.yaml` | 损失改为 BCE+Dice（无 pos_weight/fn_penalty） |

## 执行方式

### 单个实验
```bash
python run_experiment.py --config experiments/configs/baseline.yaml
```

### 换种子重跑
```bash
python run_experiment.py --config experiments/configs/baseline.yaml --seed 3407
```

### 仅评估（不训练）
```bash
python run_experiment.py --config experiments/configs/baseline.yaml --eval-only
```

### 跑完全部实验后汇总
```bash
python report.py
```

## 评估指标

- **Dice**：分割主指标
- **IoU**：交并比
- **Sensitivity**：血管检出率
- **Specificity**：背景正确率
- **Params**：模型参数量
- **Inference(ms)**：单张推理时延

## 注意事项

- 每组实验固定 seed=42，如需多种子重复请用 `--seed` 覆盖
- 所有实验共用同一份数据划分（`dataset/training` + `dataset/test`）
- 结果统一输出到 `experiments/results/<name>/metrics.json`
- `report.py` 自动汇总所有实验到 `summary.csv` 和 `summary.md`
