# Ablation Plan (8 Groups)

1. Baseline (Residual + CA + Multi-scale)
2. - Channel Attention
3. - Residual Block
4. - Multi-scale head
5. Input 256 vs 512
6. Threshold 0.4/0.5/0.6
7. CLAHE off
8. Gamma correction off

每组实验在 `experiments/configs/*.yaml` 中固化配置，训练后输出 `metrics.json`，再由 `report.py` 汇总。
