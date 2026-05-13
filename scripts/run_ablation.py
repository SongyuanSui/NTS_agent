#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import REPO_ROOT, load_config_stack, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval ablation variants.")
    parser.add_argument("--config", type=str, default="configs/experiments/ablations.yaml")
    parser.add_argument("--dataset-loader", type=str, default="ucr2015")
    parser.add_argument("--dataset", type=str, default="ECG200")
    parser.add_argument("--base-dir", type=str, default="datasets/UCR_TS_Archive_2015")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--channel-id", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--outputs-root", type=str, default="outputs")
    return parser


def _run(command: list[str]) -> int:
    return subprocess.call(command, cwd=str(REPO_ROOT))


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config_stack(args.config)
    variants = dict(cfg.get("variants", {}))
    if not variants:
        raise ValueError("No variants found in ablation config.")

    results = {}
    for name, variant in variants.items():
        retrieval_agent = str(variant.get("retrieval_agent", "retrieval_agent_hybrid"))
        save_path = Path(args.outputs_root) / "evaluations" / f"ablation_{args.dataset}_{name}.json"

        common_args = [
            "--dataset-loader",
            args.dataset_loader,
            "--dataset",
            args.dataset,
            "--base-dir",
            args.base_dir,
            "--channel-id",
            str(args.channel_id),
            "--k",
            str(args.k),
            "--save-json",
            str(save_path),
        ]
        if args.max_samples_per_split is not None:
            common_args.extend(["--max-samples-per-split", str(args.max_samples_per_split)])

        if retrieval_agent == "retrieval_agent_hybrid":
            weights = dict(variant.get("weights", {}))
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_hybrid_retrieval.py"),
                *common_args,
                "--ts-weight",
                str(weights.get("ts", 0.3333)),
                "--text-weight",
                str(weights.get("text", 0.3333)),
                "--stat-weight",
                str(weights.get("stat", 0.3334)),
            ]
        elif retrieval_agent == "retrieval_agent_stat":
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_stat_feature_retrieval.py"),
                "--dataset-loader",
                args.dataset_loader,
                "--dataset",
                args.dataset,
                "--base-dir",
                args.base_dir,
                "--channel-id",
                str(args.channel_id),
                "--k",
                str(args.k),
                "--save-json",
                str(save_path),
            ]
            if args.max_samples_per_split is not None:
                command.extend(["--max-samples-per-split", str(args.max_samples_per_split)])
        else:
            print(f"Skipping {name}: {retrieval_agent} does not have a standalone script yet.")
            continue

        print(f"=== Running ablation: {name} ===")
        code = _run(command)
        results[name] = {"exit_code": code, "result_path": str(save_path)}
        if code != 0:
            raise SystemExit(code)

    summary_path = Path(args.outputs_root) / "evaluations" / f"ablation_{args.dataset}_summary.json"
    path = save_json(summary_path, {"dataset": args.dataset, "variants": results})
    print(f"Saved ablation summary to: {path}")


if __name__ == "__main__":
    main()
