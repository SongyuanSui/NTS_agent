#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import add_dataset_args, load_dataset_from_config, print_json, resolve_config
from core.enums import TaskType
from pipelines.memory_build_pipeline import MemoryBuildPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build mixed-view memory artifacts.")
    add_dataset_args(parser)
    parser.add_argument("--task-type", type=str, default="classification")
    parser.add_argument("--channel-ids", nargs="*", type=int, default=None)
    parser.add_argument("--views", nargs="*", default=["ts", "summary", "statistic"])
    parser.add_argument("--persist-memory", action="store_true")
    parser.add_argument("--outputs-root", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default="memory_build")
    parser.add_argument("--feature-groups", nargs="*", default=None)
    parser.add_argument("--summary-style", type=str, default="statistical")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = resolve_config(args.config, "configs/data/classification.yaml")
    bundle = load_dataset_from_config(
        cfg,
        dataset_loader=args.dataset_loader,
        dataset_name=args.dataset,
        base_dir=args.base_dir,
        max_samples_per_split=args.max_samples_per_split,
    )

    components = {}
    views = {view.strip().lower() for view in args.views}
    if "ts" in views:
        from representations.raw_series import RawSeriesRepresentation

        components["ts_representation"] = RawSeriesRepresentation()
    if "summary" in views or "text" in views:
        from representations.text_summary import TextSummaryRepresentation

        components["summary_representation"] = TextSummaryRepresentation(config={"style": args.summary_style})
    if "statistic" in views or "stat" in views:
        from representations.statistics import StatisticsRepresentation

        components["statistic_representation"] = StatisticsRepresentation()

    pipeline = MemoryBuildPipeline(
        name="memory_build_pipeline",
        components=components,
        config={"task_type": args.task_type},
    )
    result = pipeline.build_memory_bank(
        samples=bundle.train.samples,
        task_type=TaskType(args.task_type),
        channel_ids=args.channel_ids,
        context={
            "persist_memory": bool(args.persist_memory),
            "dataset_name": bundle.dataset_name,
            "experiment_name": args.experiment_name,
            "outputs_root": args.outputs_root,
            "representation_metadata": {
                "statistic": {"feature_groups": args.feature_groups},
                "summary": {"style": args.summary_style},
            },
        },
    )
    print_json({"metadata": result.metadata, "artifacts": result.artifacts})


if __name__ == "__main__":
    main()
