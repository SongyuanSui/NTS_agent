#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import print_json, save_json
from evaluation.agent_metrics import average_confidence, count_agent_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved agent outputs with lightweight diagnostics.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--save-json", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    outputs = data if isinstance(data, list) else data.get("outputs", data.get("records", []))
    payload = {
        "num_outputs": len(outputs),
        "type_counts": count_agent_outputs(outputs),
        "average_confidence": average_confidence(outputs),
    }
    if args.save_json:
        path = save_json(args.save_json, payload)
        print(f"Saved agent evaluation to: {path}")
    else:
        print_json(payload)


if __name__ == "__main__":
    main()
