"""独立评估脚本：加载 checkpoint 在测试集上计算指标并写入 metrics.json。

含全测试集平均指标与 ``per_sample`` 逐张结果，并生成 ``per_sample_metrics.csv``。

Usage:
    python evaluate.py --config experiments/configs/baseline.yaml
    python evaluate.py --config experiments/configs/baseline.yaml --checkpoint checkpoints/best_model_800.pth
    python evaluate.py --config experiments/configs/baseline.yaml --no-per-sample-csv
"""

from __future__ import annotations

import argparse

import yaml

from run_experiment import evaluate as run_evaluation
from run_experiment import resolve_checkpoint_path
from run_experiment import save_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="评估模型并写入 metrics.json（含逐样本指标）")
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint 路径（默认 experiments/results/<name>/best_model.pth）",
    )
    parser.add_argument(
        "--no-per-sample-csv",
        action="store_true",
        help="不写入 per_sample_metrics.csv",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["name"]
    ckpt_path, tried = resolve_checkpoint_path(exp_name, cfg, args.checkpoint)
    if ckpt_path is None:
        print("[错误] 未找到可用 checkpoint，已按顺序尝试:")
        for p in tried:
            print(f"  - {p}")
        print("请使用 --checkpoint 指定 .pth，或在 YAML 的 evaluation.checkpoint 中填写。")
        return

    print(f"使用权重: {ckpt_path}")
    metrics = run_evaluation(cfg, ckpt_path)
    save_metrics(
        cfg,
        metrics,
        write_per_sample_csv=not args.no_per_sample_csv,
        ckpt_path=ckpt_path,
    )

    print("\n评估完成（摘要）:")
    for key in ("name", "dice", "iou", "sensitivity", "specificity", "params", "avg_inference_ms", "num_samples", "checkpoint"):
        if key in metrics:
            print(f"  {key}: {metrics[key]}")
    print(f"  per_sample: {metrics.get('num_samples', 0)} 条，详见 metrics.json / per_sample_metrics.csv")


if __name__ == "__main__":
    main()
