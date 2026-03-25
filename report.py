"""Aggregate experiment metrics into CSV and Markdown."""

import csv
import json
from pathlib import Path


RESULTS_DIR = Path("experiments/results")
CSV_PATH = RESULTS_DIR / "summary.csv"
MD_PATH = RESULTS_DIR / "summary.md"


def main() -> None:
    rows = []
    for metrics_file in sorted(RESULTS_DIR.glob("*/metrics.json")):
        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        rows.append(payload)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "dice", "sensitivity", "specificity"])
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Experiment Summary", "", "| name | dice | sensitivity | specificity |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['name']} | {row['dice']} | {row['sensitivity']} | {row['specificity']} |")
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {CSV_PATH}")
    print(f"Saved: {MD_PATH}")


if __name__ == "__main__":
    main()
