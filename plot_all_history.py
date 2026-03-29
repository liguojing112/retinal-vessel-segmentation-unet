"""项目名称：基于U-Net的视网膜血管图像分割

作者信息：李国敬

GitHub：liguojing112

创建日期：2026/3/26

许可证：MIT

多版本训练历史可视化脚本。

读取多个 history.json 并绘制损失与指标对比图，用于实验横向分析。"""





import json

import os

from pathlib import Path



import matplotlib

import matplotlib.pyplot as plt

import numpy as np



# ---- 中文字体支持（Windows 优先）----

# 让标题/标签能显示中文；若系统缺少字体会自动回退到其它字体

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]

plt.rcParams["axes.unicode_minus"] = False





def load_history(path: Path) -> dict:

    if not path.exists():

        raise FileNotFoundError(f"找不到文件: {path}")

    with path.open("r", encoding="utf-8") as f:

        return json.load(f)





def to_float_array(values) -> np.ndarray:

    arr = np.asarray(values, dtype=np.float64)

    # 若出现非数/无穷，统一转为 NaN 便于忽略

    arr[~np.isfinite(arr)] = np.nan

    return arr





def best_score_and_epoch(scores: np.ndarray):

    finite = np.isfinite(scores)

    if not np.any(finite):

        return None, None

    idx = int(np.nanargmax(scores))

    return float(scores[idx]), idx + 1  # epoch is 1-based





BASE_DIR = Path(__file__).resolve().parent

CKPT_DIR = BASE_DIR / "checkpoints"



hist_v1 = load_history(CKPT_DIR / "history.json")

hist_v2 = load_history(CKPT_DIR / "history_v2.json")

hist_v3 = load_history(CKPT_DIR / "history_v3.json")



versions = [

    ("第一次训练 (基础U-Net)", hist_v1, "tab:blue"),

    ("第二次训练 (优化损失)", hist_v2, "tab:orange"),

    ("第三次训练 (Focal Tversky)", hist_v3, "tab:green"),

]



fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col")



for idx, (title, hist, color) in enumerate(versions):

    train_loss = to_float_array(hist.get("train_loss", []))

    val_dice = to_float_array(hist.get("val_dice", []))

    val_sens = to_float_array(hist.get("val_sens", [])) if "val_sens" in hist else None



    n = int(len(train_loss))

    epochs = np.arange(1, n + 1)



    ax_loss = axes[0, idx]

    ax_loss.plot(epochs, train_loss, color=color, linewidth=2)

    ax_loss.set_title(f"{title}\nLoss", fontsize=11)

    ax_loss.set_xlabel("Epoch")

    ax_loss.set_ylabel("Loss")

    ax_loss.grid(True, alpha=0.3)



    ax_score = axes[1, idx]

    if len(val_dice) == n and n > 0:

        ax_score.plot(epochs, val_dice, color=color, linewidth=2, marker="o", markersize=3, label="Dice")

    elif len(val_dice) > 0:

        ax_score.plot(np.arange(1, len(val_dice) + 1), val_dice, color=color, linewidth=2, marker="o", markersize=3, label="Dice")



    if val_sens is not None and len(val_sens) > 0:

        ax_score.plot(np.arange(1, len(val_sens) + 1), val_sens, "--", color=color, alpha=0.6, label="Sensitivity")



    best_dice, best_epoch = best_score_and_epoch(val_dice)

    best_text = f"{best_dice:.4f} (epoch {best_epoch})" if best_dice is not None else "N/A"



    ax_score.set_title(f"Best Dice: {best_text}", fontsize=11)

    ax_score.set_xlabel("Epoch")

    ax_score.set_ylabel("Score")

    ax_score.set_ylim(0, 1)

    ax_score.grid(True, alpha=0.3)

    ax_score.legend(loc="lower right")



plt.suptitle("Training History Comparison", fontsize=14, fontweight="bold")

plt.tight_layout()



out_path = BASE_DIR / "all_training_curves.png"

plt.savefig(out_path, dpi=300, bbox_inches="tight")

print(f"已保存对比图: {out_path}")



# 默认不弹窗，避免在 Windows / CI 环境卡住；如需弹窗，将 SHOW_PLOT=1

if str(matplotlib.get_backend()).lower() != "agg" and (os.environ.get("SHOW_PLOT") == "1"):

    plt.show()

else:

    plt.close(fig)



print("\n" + "=" * 60)

print("训练结果对比")

print("=" * 60)

for title, hist, _ in versions:

    train_loss = to_float_array(hist.get("train_loss", []))

    val_dice = to_float_array(hist.get("val_dice", []))

    best_dice, best_epoch = best_score_and_epoch(val_dice)

    print(f"{title}:")

    if best_dice is None:

        print("  最佳Dice: N/A（该版本 val_dice 全为 NaN 或缺失）")

    else:

        print(f"  最佳Dice: {best_dice:.4f} (第{best_epoch}轮)")

    if len(train_loss) > 0 and np.isfinite(train_loss[-1]):

        print(f"  最终Loss: {float(train_loss[-1]):.4f}")

    else:

        print("  最终Loss: N/A")

    print(f"  总轮数: {len(train_loss)}")

    print()