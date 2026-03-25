"""独立评估脚本：加载 checkpoint 在测试集上计算指标并写入 metrics.json。

Usage:
    python evaluate.py --config experiments/configs/baseline.yaml
    python evaluate.py --config experiments/configs/baseline.yaml --checkpoint checkpoints/best_model_800.pth
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from run_experiment import ConfigurableUNet, FundusDataset, calculate_metrics, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="评估模型并写入 metrics.json")
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint 路径（默认从 experiments/results/<name>/best_model.pth）")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["name"]
    model_cfg = cfg.get("model", {})
    prep_cfg = cfg.get("preprocessing", {})
    threshold = cfg.get("evaluation", {}).get("threshold", 0.5)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path("experiments/results") / exp_name / "best_model.pth"
    if not ckpt_path.exists():
        print(f"[错误] checkpoint 不存在: {ckpt_path}")
        return

    model = ConfigurableUNet(
        use_residual=model_cfg.get("use_residual", True),
        use_channel_attention=model_cfg.get("use_channel_attention", True),
        use_multiscale=model_cfg.get("use_multiscale", True),
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    dataset_dir = cfg.get("dataset_dir", "dataset")
    val_ds = FundusDataset(
        dataset_dir,
        train=False,
        use_clahe=prep_cfg.get("use_clahe", True),
        use_gamma=prep_cfg.get("use_gamma", True),
        gamma=prep_cfg.get("gamma", 1.2),
        clahe_clip=prep_cfg.get("clahe_clip", 2.0),
        input_size=prep_cfg.get("input_size", 512),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    all_metrics: dict[str, list[float]] = {"dice": [], "iou": [], "sensitivity": [], "specificity": []}
    inference_times: list[float] = []

    with torch.no_grad():
        for images, masks in val_loader:
            start = time.perf_counter()
            outputs = model(images)
            elapsed = (time.perf_counter() - start) * 1000
            inference_times.append(elapsed)
            m = calculate_metrics(outputs.numpy()[0, 0], masks.numpy()[0, 0], threshold)
            for k in all_metrics:
                all_metrics[k].append(m[k])

    result = {
        "name": exp_name,
        "dice": round(float(np.mean(all_metrics["dice"])), 4),
        "iou": round(float(np.mean(all_metrics["iou"])), 4),
        "sensitivity": round(float(np.mean(all_metrics["sensitivity"])), 4),
        "specificity": round(float(np.mean(all_metrics["specificity"])), 4),
        "params": count_parameters(model),
        "avg_inference_ms": round(float(np.mean(inference_times)), 2),
        "threshold": threshold,
        "checkpoint": str(ckpt_path),
    }

    out_dir = Path("experiments/results") / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n评估完成: {out_path}")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
