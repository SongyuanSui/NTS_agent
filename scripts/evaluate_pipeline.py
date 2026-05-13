#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import print_json, save_json
from evaluation.anomaly_metrics import binary_anomaly_metrics
from evaluation.classification_metrics import accuracy_score, macro_f1_score
from evaluation.retrieval_metrics import compute_topk_accuracy_and_precision_at_k


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved pipeline outputs.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--task-type", choices=["classification", "anomaly_window", "retrieval"], default="retrieval")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--save-json", type=str, default=None)
    return parser


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        if data and "per_query" in data[0]:
            records = []
            for item in data:
                records.extend(item.get("per_query", []))
            return records
        return data
    if isinstance(data, dict):
        return list(data.get("per_query", data.get("records", [])))
    raise TypeError("Unsupported input JSON shape.")


def main() -> None:
    args = build_parser().parse_args()
    data = _load(args.input_json)
    records = _extract_records(data)

    if args.task_type == "retrieval":
        true_labels = [record["true_label"] for record in records]
        predicted_labels = [record["retrieved_labels"] for record in records]
        metrics = compute_topk_accuracy_and_precision_at_k(true_labels, predicted_labels, k=args.k)
    else:
        true_labels = [record.get("true_label") for record in records]
        predicted = [record.get("prediction", record.get("predicted_label")) for record in records]
        if args.task_type == "classification":
            metrics = {
                "accuracy": accuracy_score(true_labels, predicted),
                "macro_f1": macro_f1_score(true_labels, predicted),
            }
        else:
            metrics = binary_anomaly_metrics(true_labels, predicted)

    payload = {"num_records": len(records), "metrics": metrics}
    if args.save_json:
        path = save_json(args.save_json, payload)
        print(f"Saved evaluation to: {path}")
    else:
        print_json(payload)


if __name__ == "__main__":
    main()
