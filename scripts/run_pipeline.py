#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import tempfile
import yaml
import os

from common import REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a supported NTS pipeline wrapper.")
    parser.add_argument(
        "--mode",
        choices=["stat", "hybrid", "inference"],
        default="stat",
        help="Pipeline path to run: stat feature retrieval, hybrid retrieval, or full inference pipeline.",
    )
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected script.")
    # LLM overrides (will be used to create a temporary config if --config is provided downstream)
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM override when forwarding to pipeline.")
    parser.add_argument("--llm-provider", type=str, default=None, help="LLM provider override (e.g. qwen, openai).")
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model override (e.g. Qwen/Qwen2.5-72B-Instruct).")
    parser.add_argument("--llm-base-url", type=str, default=None, help="LLM base_url override for self-hosted endpoints.")
    parser.add_argument("--use-sglang", action="store_true", help="If set, mark use_sglang=true in the LLM override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "stat":
        target = "run_stat_feature_retrieval.py"
    elif args.mode == "hybrid":
        target = "run_hybrid_retrieval.py"
    else:
        target = "run_inference_pipeline.py"
    script = REPO_ROOT / "scripts" / target
    forwarded = list(args.pipeline_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    # Collect LLM override values
    llm_overrides = {}
    if args.use_llm or args.llm_provider or args.llm_model or args.llm_base_url or args.use_sglang:
        if args.llm_provider:
            llm_overrides["provider"] = args.llm_provider
        if args.llm_model:
            llm_overrides["model"] = args.llm_model
        if args.llm_base_url:
            llm_overrides["base_url"] = args.llm_base_url
        if args.use_sglang:
            llm_overrides["use_sglang"] = True

    # Helper: find downstream --config index in forwarded args
    def find_config_index(lst: list[str]) -> int:
        for i, v in enumerate(lst):
            if v == "--config":
                return i
        return -1

    config_idx = find_config_index(forwarded)

    tmp_file_to_remove: str | None = None
    if llm_overrides and config_idx != -1 and config_idx + 1 < len(forwarded):
        base_config = forwarded[config_idx + 1]
        # create temp override YAML that references the base_config as a default
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tmp:
            tmp_path = Path(tmp.name)
            # Keep backward compatibility by writing both top-level llm and
            # pipeline.params.llm override shapes.
            payload = {
                "defaults": [str(base_config)],
                "llm": llm_overrides,
                "pipeline": {"params": {"llm": llm_overrides}},
            }
            yaml.safe_dump(payload, tmp)
            tmp_file_to_remove = str(tmp_path)

        # replace the downstream --config path with our temporary file
        forwarded[config_idx + 1] = tmp_file_to_remove
    elif llm_overrides and config_idx == -1:
        # No downstream --config to base on; set environment vars as a fallback
        if args.llm_provider:
            os.environ.setdefault("LLM_PROVIDER", args.llm_provider)
        if args.llm_model:
            os.environ.setdefault("LLM_MODEL", args.llm_model)
        if args.llm_base_url:
            os.environ.setdefault("LLM_BASE_URL", args.llm_base_url)
        if args.use_sglang:
            os.environ.setdefault("LLM_USE_SGLANG", "1")

    command = [sys.executable, str(script), *forwarded]

    # Run the child process and ensure temporary config file is removed afterwards.
    try:
        rc = subprocess.call(command, cwd=str(REPO_ROOT))
    finally:
        if tmp_file_to_remove:
            try:
                Path(tmp_file_to_remove).unlink()
            except Exception:
                # do not fail the wrapper if cleanup fails
                pass

    raise SystemExit(rc)


if __name__ == "__main__":
    main()
