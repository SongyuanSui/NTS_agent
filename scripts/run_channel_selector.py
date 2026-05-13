#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import add_dataset_args, load_dataset_from_config, print_json, resolve_config, save_json, task_spec_from_config
from agents.channel_selector import ChannelSelectorAgent
from agents.schemas import ChannelSelectorInput
from core.enums import TaskType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dataset-level channel selection.")
    add_dataset_args(parser)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--nn-eval-samples", type=int, default=None)
    parser.add_argument("--diversity-threshold", type=float, default=None)
    parser.add_argument("--save-json", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = resolve_config(args.config, "configs/agents/channel_selector.yaml")
    data_cfg = resolve_config("configs/data/classification.yaml", "configs/data/classification.yaml")
    cfg = {**data_cfg, **cfg}

    bundle = load_dataset_from_config(
        cfg,
        dataset_loader=args.dataset_loader,
        dataset_name=args.dataset,
        base_dir=args.base_dir,
        max_samples_per_split=args.max_samples_per_split,
    )
    task_spec = task_spec_from_config(cfg, TaskType.CLASSIFICATION)
    params = dict(cfg.get("agent", {}).get("params", {}))

    top_k = int(args.top_k if args.top_k is not None else params.get("top_k", 3))
    max_len = args.max_len if args.max_len is not None else params.get("max_len")
    alpha = float(args.alpha if args.alpha is not None else params.get("alpha", 0.5))
    nn_eval_samples = int(
        args.nn_eval_samples if args.nn_eval_samples is not None else params.get("nn_eval_samples", 200)
    )
    diversity = (
        args.diversity_threshold
        if args.diversity_threshold is not None
        else params.get("diversity_threshold")
    )

    agent = ChannelSelectorAgent(name="channel_selector", config=params)
    output = agent.run(
        ChannelSelectorInput(
            train_samples=bundle.train.samples,
            task_spec=task_spec,
            top_k=top_k,
            max_len=max_len,
            z_norm=bool(params.get("z_norm", True)),
            alpha=alpha,
            nn_eval_samples=nn_eval_samples,
            diversity_threshold=diversity,
            random_seed=int(params.get("random_seed", 42)),
        )
    )

    payload = {
        "dataset": bundle.dataset_name,
        "selected_channel_ids": output.selected_channel_ids,
        "ranked_channel_ids": output.ranked_channel_ids,
        "channel_scores": output.channel_scores,
        "metadata": output.metadata,
    }
    if args.save_json:
        path = save_json(args.save_json, payload)
        print(f"Saved channel selection to: {path}")
    else:
        print_json(payload)


if __name__ == "__main__":
    main()
