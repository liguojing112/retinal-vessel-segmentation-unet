"""从 ``per_sample_metrics.csv`` 生成统计摘要、排序表与 JSON 汇总。

Usage:
    python experiments/summarize_per_sample_csv.py \\
        --input experiments/results/loss_bce_dice/per_sample_metrics.csv

    # 摘要里「Dice 最高 / 最低」各列多少张（默认 20，可用 --top-n 改）
    python experiments/summarize_per_sample_csv.py \\
        --input experiments/results/loss_bce_dice/per_sample_metrics.csv --top-n 30
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


NUMERIC_KEYS = ("dice", "iou", "sensitivity", "specificity", "inference_ms")


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rows_with_floats(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        item: dict[str, Any] = {"filename": r["filename"]}
        for k in NUMERIC_KEYS:
            item[k] = float(r[k])
        out.append(item)
    return out


def aggregate_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    n = len(values)
    out: dict[str, float] = {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if n > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }
    try:
        qs = statistics.quantiles(values, n=4, method="inclusive")
        out["q1"] = float(qs[0])
        out["q3"] = float(qs[2])
    except statistics.StatisticsError:
        out["q1"] = out["median"]
        out["q3"] = out["median"]
    return out


def write_sorted_csv(rows: list[dict[str, Any]], path: Path, sort_key: str, descending: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda x: x[sort_key], reverse=descending)
    fieldnames = ["filename", *NUMERIC_KEYS]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r[k] for k in fieldnames})


def write_summary_md(
    rows: list[dict[str, Any]],
    stats_by_metric: dict[str, dict[str, float]],
    path: Path,
    title: str,
    top_n: int,
) -> None:
    by_dice = sorted(rows, key=lambda x: x["dice"], reverse=True)
    worst = sorted(rows, key=lambda x: x["dice"])[:top_n]

    lines = [
        f"# {title}",
        "",
        f"测试样本数：**{len(rows)}**",
        "",
        "## 各指标统计（0~1 为小数；inference_ms 为毫秒）",
        "",
        "| 指标 | 均值 | 标准差 | 最小 | 最大 | 中位数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, st in stats_by_metric.items():
        label = metric if metric != "inference_ms" else "推理耗时(ms)"
        lines.append(
            f"| {label} | {st['mean']:.4f} | {st['std']:.4f} | {st['min']:.4f} | "
            f"{st['max']:.4f} | {st['median']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"## Dice 最高的 {top_n} 张",
            "",
            "| 文件名 | Dice | IoU | Sens | Spec | 推理(ms) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in by_dice[:top_n]:
        lines.append(
            f"| {r['filename']} | {r['dice']:.4f} | {r['iou']:.4f} | {r['sensitivity']:.4f} | "
            f"{r['specificity']:.4f} | {r['inference_ms']:.2f} |"
        )

    lines.extend(
        [
            "",
            f"## Dice 最低的 {top_n} 张（可重点复查）",
            "",
            "| 文件名 | Dice | IoU | Sens | Spec | 推理(ms) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in worst:
        lines.append(
            f"| {r['filename']} | {r['dice']:.4f} | {r['iou']:.4f} | {r['sensitivity']:.4f} | "
            f"{r['specificity']:.4f} | {r['inference_ms']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 生成文件说明",
            "",
            "- `per_sample_sorted_by_dice.csv`：全部样本按 Dice 降序",
            "- `per_sample_stats.json`：各指标聚合统计（机器可读）",
            "- 摘要中高低分表格行数由运行参数 `--top-n` 控制（本文件生成时使用的 N 见上文章节标题）",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="整理 per_sample_metrics.csv")
    parser.add_argument("--input", type=Path, required=True, help="输入 CSV 路径")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录（默认与输入 CSV 同目录）",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        metavar="N",
        help="摘要 Markdown 中 Dice 最高与最低各列出 N 张（默认 20）",
    )
    args = parser.parse_args()

    csv_path = args.input.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"文件不存在: {csv_path}")

    out_dir = (args.out_dir or csv_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_rows(csv_path)
    rows = rows_with_floats(raw)

    stats_by_metric: dict[str, dict[str, float]] = {}
    for key in NUMERIC_KEYS:
        stats_by_metric[key] = aggregate_stats([r[key] for r in rows])

    title = f"逐样本指标整理 — {csv_path.parent.name}"
    write_summary_md(rows, stats_by_metric, out_dir / "per_sample_summary.md", title, args.top_n)
    write_sorted_csv(rows, out_dir / "per_sample_sorted_by_dice.csv", "dice", descending=True)

    json_path = out_dir / "per_sample_stats.json"
    json_path.write_text(
        json.dumps(
            {
                "source_csv": str(csv_path),
                "num_samples": len(rows),
                "metrics": stats_by_metric,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"已写入: {out_dir / 'per_sample_summary.md'}")
    print(f"已写入: {out_dir / 'per_sample_sorted_by_dice.csv'}")
    print(f"已写入: {json_path}")


if __name__ == "__main__":
    main()
