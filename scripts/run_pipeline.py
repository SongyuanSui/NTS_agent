#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a supported NTS pipeline wrapper.")
    parser.add_argument(
        "--mode",
        choices=["stat", "hybrid"],
        default="stat",
        help="Pipeline path to run: stat feature retrieval or hybrid retrieval.",
    )
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected script.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target = "run_stat_feature_retrieval.py" if args.mode == "stat" else "run_hybrid_retrieval.py"
    script = REPO_ROOT / "scripts" / target
    forwarded = list(args.pipeline_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    command = [sys.executable, str(script), *forwarded]
    raise SystemExit(subprocess.call(command, cwd=str(REPO_ROOT)))


if __name__ == "__main__":
    main()
