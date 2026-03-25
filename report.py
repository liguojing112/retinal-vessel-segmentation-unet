"""汇总所有实验指标到 CSV 和 Markdown 表格。

Usage:
    python report.py
"""

import csv
import json
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
CSV_PATH = RESULTS_DIR / "summary.csv"
MD_PATH = RESULTS_DIR / "summary.md"

COLUMNS = ["name", "dice", "iou", "sensitivity", "specificity", "params", "avg_inference_ms"]
DISPLAY_HEADERS = {
    "name": "实验名称",
    "dice": "Dice",
    "iou": "IoU",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "params": "参数量",
    "avg_inference_ms": "推理(ms)",
}


def main() -> None:
    rows: list[dict] = []
    for metrics_file in sorted(RESULTS_DIR.glob("*/metrics.json")):
        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        row: dict = {}
        for col in COLUMNS:
            row[col] = payload.get(col, "-")
        rows.append(row)

    if not rows:
        print("未找到任何 metrics.json")
        return

    rows.sort(key=lambda r: r.get("dice", 0) if isinstance(r.get("dice"), (int, float)) else 0, reverse=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    header_line = "| " + " | ".join(DISPLAY_HEADERS.get(c, c) for c in COLUMNS) + " |"
    sep_line = "|" + "|".join("---:" if c != "name" else "---" for c in COLUMNS) + "|"
    md_lines = ["# Experiment Summary", "", header_line, sep_line]
    for row in rows:
        cells: list[str] = []
        for col in COLUMNS:
            val = row[col]
            if col == "params" and isinstance(val, int):
                cells.append(f"{val:,}")
            elif isinstance(val, float):
                cells.append(f"{val}")
            else:
                cells.append(str(val))
        md_lines.append("| " + " | ".join(cells) + " |")

    MD_PATH.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"已生成: {CSV_PATH}")
    print(f"已生成: {MD_PATH}")
    print(f"共 {len(rows)} 条实验记录（按 Dice 降序排列）")


if __name__ == "__main__":
    main()
