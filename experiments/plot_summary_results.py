"""将 ``experiments/results/summary.csv`` 绘制成 PNG（柱状对比 + 表格，与 summary.md 数据一致）。

Usage:
    python experiments/plot_summary_results.py
    python experiments/plot_summary_results.py --input experiments/results/summary.csv --output experiments/results/summary_visualization.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


METRIC_LABELS: dict[str, str] = {
    "name": "实验名称",
    "dice": "Dice",
    "iou": "IoU",
    "sensitivity": "敏感度",
    "specificity": "特异性",
    "params": "参数量",
    "avg_inference_ms": "推理(ms)",
}

# 适合画在同一组柱状图里的指标（0~1 或同量级）；其余单独子图
BAR_METRIC_KEYS = ("dice", "iou", "sensitivity", "specificity")


def configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def load_summary_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV 无数据行: {csv_path}")
    if "name" not in fieldnames:
        raise ValueError("CSV 必须包含 name 列")
    return fieldnames, rows


def parse_float_cell(value: str) -> float | None:
    value = (value or "").strip()
    if value in ("", "-", "—"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def pick_bar_metrics(fieldnames: list[str], sample: dict[str, str]) -> list[str]:
    """选择用于分组柱状图的列（BAR_METRIC_KEYS 中且可解析为浮点）。"""
    out: list[str] = []
    for key in BAR_METRIC_KEYS:
        if key not in fieldnames:
            continue
        if parse_float_cell(sample.get(key, "")) is not None:
            out.append(key)
    return out


def plot_summary(
    csv_path: Path,
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    fieldnames, rows = load_summary_rows(csv_path)
    names = [r["name"] for r in rows]
    bar_keys = pick_bar_metrics(fieldnames, rows[0])

    fig = plt.figure(figsize=(max(10, len(names) * 2.2), 10))
    gs = GridSpec(2, 1, height_ratios=[2.2, 1.0], hspace=0.38)
    ax_bar = fig.add_subplot(gs[0])

    if bar_keys:
        x = np.arange(len(names))
        n_m = len(bar_keys)
        width = min(0.8 / max(n_m, 1), 0.22)
        offsets = (np.arange(n_m) - (n_m - 1) / 2) * width * 1.15
        bar_colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948"]

        for i, key in enumerate(bar_keys):
            vals: list[float] = []
            for r in rows:
                v = parse_float_cell(r.get(key, ""))
                vals.append(v if v is not None else 0.0)
            ax_bar.bar(
                x + offsets[i],
                vals,
                width * 1.05,
                label=METRIC_LABELS.get(key, key),
                color=bar_colors[i % len(bar_colors)],
                edgecolor="white",
                linewidth=0.8,
            )
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(names, rotation=25, ha="right")
        ax_bar.set_ylabel("指标值")
        ax_bar.set_title("实验汇总对比（summary.csv）", fontsize=13)
        ax_bar.legend(loc="upper right", fontsize=9)
        ax_bar.set_ylim(0, max(1.05, ax_bar.get_ylim()[1]))
        ax_bar.grid(True, axis="y", linestyle="--", alpha=0.35)
    else:
        ax_bar.text(0.5, 0.5, "无可绘制的数值列（dice/iou/sensitivity/specificity）", ha="center", va="center")
        ax_bar.axis("off")

    # 表格：与 summary.md 一致，展示全部列
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    display_cols = [c for c in fieldnames if c]
    headers = [METRIC_LABELS.get(c, c) for c in display_cols]
    cell_text: list[list[str]] = []
    for r in rows:
        row_cells: list[str] = []
        for c in display_cols:
            row_cells.append(str(r.get(c, "")).strip())
        cell_text.append(row_cells)

    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472c4")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f0f2f5" if row % 2 else "white")

    fig.suptitle(f"数据来源: {csv_path.name}", fontsize=10, y=0.02, color="gray")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="summary.csv / summary.md 数据可视化 PNG")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/results/summary.csv"),
        help="summary.csv 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/summary_visualization.png"),
        help="输出 PNG 路径",
    )
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    csv_path = args.input.expanduser()
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    else:
        csv_path = csv_path.resolve()

    if not csv_path.is_file():
        raise SystemExit(f"找不到文件: {csv_path}\n请先运行 python report.py 生成 summary.csv")

    out_path = args.output.expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    else:
        out_path = out_path.resolve()

    configure_matplotlib()
    plot_summary(csv_path, out_path, args.dpi)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
