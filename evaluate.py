"""Simple evaluation utility for experiment closure.

Usage:
    python evaluate.py --name baseline --dice 0.865 --sensitivity 0.82 --specificity 0.975
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--dice", type=float, required=True)
    parser.add_argument("--sensitivity", type=float, required=True)
    parser.add_argument("--specificity", type=float, required=True)
    args = parser.parse_args()

    out_dir = Path("experiments/results") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": args.name,
        "dice": args.dice,
        "sensitivity": args.sensitivity,
        "specificity": args.specificity,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
