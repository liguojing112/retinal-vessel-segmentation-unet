"""一次性绘制所有实验的验证集 Dice 曲线，用于消融对比。

默认从本目录下的 ``experiments/results/*/history.json`` 读取数据
（与从云服务器同步到 ``results/`` 的常见目录结构一致）。

用法::

    cd results
    pip install matplotlib
    python plot_ablation_curves.py

    # 指定结果根目录（仓库内标准路径）
    python plot_ablation_curves.py --root ../experiments/results

    # 突出 baseline 曲线
    python plot_ablation_curves.py --highlight baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _find_history_files(root: Path) -> list[Path]:
    """在 root 下查找各实验子目录中的 history.json。"""
    if not root.is_dir():
        return []
    return sorted(root.glob("*/history.json"))


def _load_val_dice(path: Path) -> tuple[str, list[float]]:
    """读取 history.json，返回 (实验名, val_dice 列表)。"""
    name = path.parent.name
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("val_dice")
    if raw is None:
        raise KeyError(f"{path}: 缺少 val_dice 字段")
    series = [float(x) for x in raw]
    return name, series


def _configure_matplotlib_chinese() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_all_curves(
    root: Path,
    output: Path,
    highlight: str | None,
    as_percent: bool,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    files = _find_history_files(root)
    if not files:
        print(f"未找到任何 history.json，请检查目录: {root.resolve()}", file=sys.stderr)
        sys.exit(1)

    series_list: list[tuple[str, list[float]]] = []
    for fp in files:
        try:
            series_list.append(_load_val_dice(fp))
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"跳过 {fp}: {exc}", file=sys.stderr)

    if not series_list:
        print("没有可用的 val_dice 数据。", file=sys.stderr)
        sys.exit(1)

    # 按最终 epoch 的 Dice 降序，图例更易读
    series_list.sort(key=lambda item: item[1][-1] if item[1] else 0.0, reverse=True)

    _configure_matplotlib_chinese()
    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, (name, values) in enumerate(series_list):
        y = np.asarray(values, dtype=np.float64)
        if as_percent:
            y = y * 100.0
        x = np.arange(1, len(y) + 1)
        color = f"C{idx % 10}"
        lw = 2.8 if highlight and name == highlight else 1.6
        alpha = 1.0 if highlight and name == highlight else 0.85
        zorder = 10 if highlight and name == highlight else 1
        ax.plot(x, y, label=name, color=color, linewidth=lw, alpha=alpha, zorder=zorder)

    ax.set_xlabel("Epoch", fontsize=12)
    ylab = "验证集 Dice (%)" if as_percent else "验证集 Dice"
    ax.set_ylabel(ylab, fontsize=12)
    ax.set_title("消融实验 — 验证集 Dice 曲线对比", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制所有实验的 val_dice 曲线对比图")
    default_root = Path(__file__).resolve().parent / "experiments" / "results"
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"实验结果根目录（其下为 <实验名>/history.json），默认: {default_root}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "ablation_val_dice_curves.png",
        help="输出图片路径",
    )
    parser.add_argument(
        "--highlight",
        type=str,
        default=None,
        help="加粗高亮的实验子目录名，例如 baseline",
    )
    parser.add_argument(
        "--raw-scale",
        action="store_true",
        help="纵轴使用 0~1 原始刻度（默认乘以 100 显示为百分比）",
    )
    parser.add_argument("--dpi", type=int, default=160, help="输出 PNG 分辨率")
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("请先安装: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    plot_all_curves(
        root=args.root.resolve(),
        output=args.output.resolve(),
        highlight=args.highlight,
        as_percent=not args.raw_scale,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
