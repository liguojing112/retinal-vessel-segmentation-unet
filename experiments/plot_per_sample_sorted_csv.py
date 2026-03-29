"""将 ``per_sample_sorted_by_dice.csv``（或同结构 CSV）绘制成多子图 PNG。

**注意**：`--input` 必须写**真实 CSV 路径**，不能把文档里的占位符 ``...`` 原样粘贴。

示例（在仓库根目录执行）::

    python experiments/plot_per_sample_sorted_csv.py ^
        --input experiments/results/loss_bce_dice/per_sample_sorted_by_dice.csv

输出到 D 盘（目录不存在会自动创建）::

    python experiments/plot_per_sample_sorted_csv.py ^
        --input experiments/results/loss_bce_dice/per_sample_sorted_by_dice.csv ^
        --output D:/out/my_plot.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def load_numeric_columns(csv_path: Path) -> dict[str, list[float]]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV 无数据行: {csv_path}")
    keys = ("dice", "iou", "sensitivity", "specificity", "inference_ms")
    out: dict[str, list[float]] = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            out[k].append(float(row[k]))
    return out


def configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_figure(
    data: dict[str, list[float]],
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    dice = np.asarray(data["dice"], dtype=np.float64)
    iou = np.asarray(data["iou"], dtype=np.float64)
    sens = np.asarray(data["sensitivity"], dtype=np.float64)
    spec = np.asarray(data["specificity"], dtype=np.float64)
    ms = np.asarray(data["inference_ms"], dtype=np.float64)
    n = len(dice)
    ranks = np.arange(1, n + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(title, fontsize=14, y=1.02)

    ax0 = axes[0, 0]
    ax0.hist(dice, bins=min(30, max(10, n // 5)), color="steelblue", edgecolor="white", alpha=0.85)
    ax0.axvline(float(np.mean(dice)), color="darkred", linestyle="--", linewidth=2, label=f"均值 {np.mean(dice):.4f}")
    ax0.set_xlabel("Dice")
    ax0.set_ylabel("样本数")
    ax0.set_title("Dice 分布直方图")
    ax0.legend(loc="upper left")

    ax1 = axes[0, 1]
    bp = ax1.boxplot(
        [dice, iou, sens, spec],
        tick_labels=["Dice", "IoU", "Sens", "Spec"],
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightcyan")
        patch.set_alpha(0.85)
    ax1.set_ylabel("取值 (0~1)")
    ax1.set_title("分割指标箱线图")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)

    ax2 = axes[1, 0]
    ax2.plot(ranks, dice, color="teal", linewidth=1.2, marker=".", markersize=3)
    ax2.fill_between(ranks, dice, alpha=0.15, color="teal")
    ax2.set_xlabel("排序位次（1 = Dice 最高）")
    ax2.set_ylabel("Dice")
    ax2.set_title("按 Dice 降序：全测试集曲线")
    ax2.set_xlim(1, n)
    ax2.grid(True, linestyle="--", alpha=0.35)

    ax3 = axes[1, 1]
    sc = ax3.scatter(dice, ms, c=iou, cmap="viridis", alpha=0.65, s=28, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)
    cb.set_label("IoU")
    ax3.set_xlabel("Dice")
    ax3.set_ylabel("推理耗时 (ms)")
    ax3.set_title("Dice vs 推理耗时（颜色 = IoU）")
    ax3.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _exit_missing_csv(csv_path: Path, raw_input: str) -> None:
    """打印明确错误说明后退出。"""
    lines = [
        f"[错误] 找不到 CSV 文件: {csv_path}",
        "",
        "常见原因：",
        "  1) 把说明里的「...」当成路径粘贴了——请换成下面这样的完整相对或绝对路径。",
        "  2) 当前工作目录不是仓库根目录——请先 cd 到 retinal-vessel-segmentation-unet 再执行。",
        "",
        "在仓库根目录可复制执行（整行一条）：",
        r'  python experiments/plot_per_sample_sorted_csv.py --input experiments/results/loss_bce_dice/per_sample_sorted_by_dice.csv',
        "",
        f"您本次传入的 --input 原始参数为: {raw_input!r}",
    ]
    raise SystemExit("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="逐样本 CSV 可视化 PNG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python experiments/plot_per_sample_sorted_csv.py "
        "--input experiments/results/loss_bce_dice/per_sample_sorted_by_dice.csv",
    )
    parser.add_argument(
        "--input",
        dest="input_csv",
        type=Path,
        required=True,
        metavar="CSV",
        help="CSV 文件路径（不要用 ... 占位；需含 dice/iou 等列）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 PNG（默认与 CSV 同目录 per_sample_visualization.png；父目录不存在会自动创建）",
    )
    parser.add_argument("--dpi", type=int, default=160, help="输出分辨率")
    args = parser.parse_args()

    raw_in = str(args.input_csv)
    if raw_in.strip() in ("...", "…", ".", ".."):
        _exit_missing_csv(Path(raw_in), raw_in)

    csv_path = args.input_csv.expanduser()
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    else:
        csv_path = csv_path.resolve()

    if csv_path.is_dir():
        raise SystemExit(
            f"[错误] --input 指向了目录而不是文件: {csv_path}\n"
            "请指向具体 .csv，例如: .../per_sample_sorted_by_dice.csv"
        )
    if not csv_path.is_file():
        _exit_missing_csv(csv_path, raw_in)

    out_path = args.output
    if out_path is None:
        out_path = csv_path.parent / "per_sample_visualization.png"
    else:
        out_path = out_path.expanduser()
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        else:
            out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    data = load_numeric_columns(csv_path)
    title = f"逐样本指标可视化 — {csv_path.parent.name}"
    plot_figure(data, title, out_path, args.dpi)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
