#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.constants import DEFAULT_OUTPUTS_ROOT
from core.enums import TaskType
from data.dataset_registry import get_dataset_loader, list_dataset_loaders
from data.loaders.classification_multivariate_loader import DEFAULT_UEA_DIR
from data.loaders.classification_univariate_loader import DEFAULT_UCR2015_DIR
from pipelines.stat_feature_retrieval_pipeline import (
    MemoryPersistenceConfig,
    StatFeatureRetrievalPipeline,
)
from representations.statistics import StatisticsRepresentation
from retrieval.stat_retrievers import StatKNNRetriever


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run stat-feature retrieval on classification datasets (e.g., UCR2015, UEA).",
    )
    parser.add_argument(
        "--dataset-loader",
        type=str,
        default="ucr2015",
        choices=list_dataset_loaders(),
        help="Dataset loader key from registry (e.g., ucr2015, uea)",
    )
    parser.add_argument("--dataset", type=str, default="ECG200", help="Dataset name under the selected loader")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Optional dataset root directory override; if omitted, loader default is used",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional cap for train/test samples per split",
    )
    parser.add_argument("--distance", type=str, default="cosine", choices=["cosine", "l2", "weighted_l2"])
    parser.add_argument("--normalize", type=str, default="zscore", choices=["none", "zscore", "robust", "log1p_robust"])
    parser.add_argument("--k", type=int, default=1, help="Top-k for retrieval metrics")
    parser.add_argument(
        "--channel-id",
        type=int,
        default=0,
        help="Channel index used for retrieval/evaluation (for multivariate datasets)",
    )
    parser.add_argument(
        "--feature-groups",
        nargs="*",
        default=None,
        help="Optional feature group names passed to StatisticsRepresentation",
    )
    parser.add_argument(
        "--run-default-grid",
        action="store_true",
        help="Run 4 default experiments: (cosine,zscore,k=1/5), (l2,robust,k=1/5)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save experiment results as JSON",
    )
    parser.add_argument(
        "--outputs-root",
        type=str,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Outputs root used when --save-json is not provided",
    )
    parser.add_argument(
        "--persist-memory",
        action="store_true",
        help="Persist built memory artifacts under outputs/memory/<dataset>_<experiment>",
    )
    parser.add_argument(
        "--no-reuse-memory",
        action="store_true",
        help="Disable loading existing compatible memory artifacts before rebuilding.",
    )
    parser.add_argument(
        "--force-rebuild-memory",
        action="store_true",
        help="Force rebuilding memory artifacts even when compatible artifacts already exist.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="stat_feature_retrieval",
        help="Experiment name used for memory artifact directory naming",
    )
    return parser


def _default_base_dir_for_loader(loader_name: str) -> str:
    normalized = loader_name.strip().lower()
    if normalized == "ucr2015":
        return str(DEFAULT_UCR2015_DIR)
    if normalized == "uea":
        return str(DEFAULT_UEA_DIR)
    raise KeyError(f"No default base-dir configured for loader '{loader_name}'")


def _run_once(
    train_samples,
    test_samples,
    dataset_name: str,
    experiment_name: str,
    outputs_root: str,
    persist_memory: bool,
    reuse_memory: bool,
    force_rebuild_memory: bool,
    distance: str,
    normalize: str,
    k: int,
    channel_id: int,
    feature_groups: list[str] | None,
) -> dict[str, Any]:
    pipeline = StatFeatureRetrievalPipeline(
        components={
            "representation": StatisticsRepresentation(),
            "retriever": StatKNNRetriever(
                config={
                    "distance": distance,
                    "normalize": normalize,
                }
            ),
        },
        config={
            "top_k": k,
            "task_type": TaskType.CLASSIFICATION,
            "representation_metadata": {
                "feature_groups": feature_groups,
            },
        },
    )

    result = pipeline.evaluate_split(
        train_samples=train_samples,
        query_samples=test_samples,
        task_type=TaskType.CLASSIFICATION,
        top_k=k,
        channel_id=channel_id,
        memory_bank=None,
        memory_config=MemoryPersistenceConfig(
            persist_memory=bool(persist_memory),
            reuse_memory=bool(reuse_memory),
            force_rebuild=bool(force_rebuild_memory),
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            outputs_root=outputs_root,
            selected_channel_ids=[channel_id],
        ),
    )

    return result


def _default_result_path(outputs_root: str, dataset: str) -> Path:
    return Path(outputs_root) / "evaluations" / f"stat_retrieval_{dataset}.json"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be a positive integer")
    if args.channel_id < 0:
        raise ValueError("--channel-id must be non-negative")

    loader = get_dataset_loader(args.dataset_loader)
    base_dir = args.base_dir or _default_base_dir_for_loader(args.dataset_loader)
    bundle = loader.load(
        dataset_name=args.dataset,
        base_dir=base_dir,
        remap_labels=True,
        max_samples_per_split=args.max_samples_per_split,
    )

    train_samples = bundle.train.samples
    test_samples = bundle.test.samples

    if not train_samples:
        raise ValueError("Loaded empty train split")
    if not test_samples:
        raise ValueError("Loaded empty test split")

    n_channels = int(train_samples[0].num_channels)
    if args.channel_id >= n_channels:
        raise ValueError(
            f"--channel-id out of range: {args.channel_id} for dataset with {n_channels} channels"
        )

    if args.run_default_grid:
        configs = [
            {"distance": "cosine", "normalize": "zscore", "k": 1},
            {"distance": "cosine", "normalize": "zscore", "k": 5},
            {"distance": "l2", "normalize": "robust", "k": 1},
            {"distance": "l2", "normalize": "robust", "k": 5},
        ]
    else:
        configs = [
            {"distance": args.distance, "normalize": args.normalize, "k": args.k},
        ]

    all_results: list[dict[str, Any]] = []

    for idx, cfg in enumerate(configs, start=1):
        print(
            f"=== Test {idx}: distance={cfg['distance']} normalize={cfg['normalize']} k={cfg['k']} ==="
        )

        exp_name = args.experiment_name
        if args.run_default_grid:
            exp_name = (
                f"{args.experiment_name}_"
                f"dist-{cfg['distance']}_norm-{cfg['normalize']}_k-{cfg['k']}"
            )

        result = _run_once(
            train_samples=train_samples,
            test_samples=test_samples,
            dataset_name=args.dataset,
            experiment_name=exp_name,
            outputs_root=args.outputs_root,
            persist_memory=bool(args.persist_memory),
            reuse_memory=not bool(args.no_reuse_memory),
            force_rebuild_memory=bool(args.force_rebuild_memory),
            distance=cfg["distance"],
            normalize=cfg["normalize"],
            k=cfg["k"],
            channel_id=args.channel_id,
            feature_groups=args.feature_groups,
        )
        result["dataset"] = args.dataset
        result["dataset_loader"] = args.dataset_loader
        result["channel_id"] = args.channel_id
        result["distance"] = cfg["distance"]
        result["normalize"] = cfg["normalize"]
        all_results.append(result)

        metrics = result["metrics"]
        print(
            "top_k_accuracy={:.4f}, precision_at_k={:.4f}, train={}, query={}".format(
                metrics["top_k_accuracy"],
                metrics["precision_at_k"],
                result["num_train"],
                result["num_query"],
            )
        )
        if "memory_artifacts" in result:
            print("memory artifacts:")
            for key, value in result["memory_artifacts"].items():
                print(f"  - {key}: {value}")
        print()

    save_path = Path(args.save_json) if args.save_json else _default_result_path(args.outputs_root, args.dataset)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(all_results, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Saved results to: {save_path}")


if __name__ == "__main__":
    main()
