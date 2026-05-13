#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys

from common import REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a supported standalone agent.")
    parser.add_argument(
        "--agent",
        choices=["channel_selector"],
        default="channel_selector",
        help="Agent entrypoint to run.",
    )
    parser.add_argument("agent_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the agent script.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.agent == "channel_selector":
        script = REPO_ROOT / "scripts" / "run_channel_selector.py"
    else:
        raise ValueError(f"Unsupported agent: {args.agent}")

    forwarded = list(args.agent_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    raise SystemExit(subprocess.call([sys.executable, str(script), *forwarded], cwd=str(REPO_ROOT)))


if __name__ == "__main__":
    main()
