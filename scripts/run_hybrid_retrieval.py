#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

from common import add_dataset_args, load_dataset_from_config, print_json, resolve_config, save_json
from core.enums import TaskType
from core.schemas import QueryInstance, TaskSpec
from evaluation.retrieval_metrics import compute_topk_accuracy_and_precision_at_k
from pipelines.memory_build_pipeline import MemoryBuildPipeline
from representations.schemas import RepresentationInput


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mixed TS/text/stat hybrid retrieval.")
    add_dataset_args(parser)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--channel-id", type=int, default=0)
    parser.add_argument("--ts-weight", type=float, default=0.30)
    parser.add_argument("--text-weight", type=float, default=0.20)
    parser.add_argument("--stat-weight", type=float, default=0.50)
    parser.add_argument("--ts-distance", type=str, default="l2")
    parser.add_argument("--text-method", type=str, default="embedding")
    parser.add_argument("--stat-distance", type=str, default="cosine")
    parser.add_argument("--stat-normalize", type=str, default="zscore")
    parser.add_argument("--feature-groups", nargs="*", default=None)
    # LLM config for text summarization
    parser.add_argument("--llm-provider", type=str, default="qwen")
    parser.add_argument("--llm-base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-max-tokens", type=int, default=512)
    parser.add_argument("--save-json", type=str, default=None)
    return parser


def build_llm_context(args: argparse.Namespace) -> dict[str, Any]:
    from core.factories import build_llm_client
    llm_cfg = {
        "provider": args.llm_provider,
        "base_url": args.llm_base_url,
        "model": args.llm_model,
        "temperature": args.llm_temperature,
        "max_tokens": args.llm_max_tokens,
    }
    return {"llm_client": build_llm_client(llm_cfg)}


def compute_query_metadata(
    sample: Any,
    channel_id: int,
    feature_groups: list[str] | None,
    llm_context: dict[str, Any],
) -> dict[str, Any]:
    from representations.raw_series import RawSeriesRepresentation
    from representations.statistics import StatisticsRepresentation
    from representations.text_summary import TextSummaryRepresentation

    ts_output = RawSeriesRepresentation().run(RepresentationInput(samples=[sample], channel_id=channel_id))
    summary_output = TextSummaryRepresentation().run(
        RepresentationInput(samples=[sample], channel_id=channel_id),
        context=llm_context,
    )
    stat_output = StatisticsRepresentation().run(
        RepresentationInput(samples=[sample], channel_id=channel_id, metadata={"feature_groups": feature_groups})
    )
    return {
        "ts_view": ts_output.records[0].payload,
        "summary_view": summary_output.records[0].payload,
        "statistic_view": stat_output.records[0].payload,
        "channel_id": channel_id,
    }


def main() -> None:
    args = build_parser().parse_args()
    from representations.raw_series import RawSeriesRepresentation
    from representations.statistics import StatisticsRepresentation
    from representations.text_summary import TextSummaryRepresentation
    from retrieval.hybrid_retriever import WeightedHybridRetriever
    from retrieval.raw_retrievers import RawSeriesRetriever
    from retrieval.stat_retrievers import StatKNNRetriever
    from retrieval.text_retrievers import TextSummaryRetriever

    cfg = resolve_config(args.config, "configs/data/classification.yaml")
    bundle = load_dataset_from_config(
        cfg,
        dataset_loader=args.dataset_loader,
        dataset_name=args.dataset,
        base_dir=args.base_dir,
        max_samples_per_split=args.max_samples_per_split,
    )

    llm_context = build_llm_context(args)

    memory_pipeline = MemoryBuildPipeline(
        components={
            "ts_representation": RawSeriesRepresentation(),
            "summary_representation": TextSummaryRepresentation(),
            "statistic_representation": StatisticsRepresentation(),
        },
        config={"task_type": TaskType.CLASSIFICATION},
    )
    memory = memory_pipeline.build_memory_bank(
        samples=bundle.train.samples,
        task_type=TaskType.CLASSIFICATION,
        channel_ids=[args.channel_id],
        context={
            **llm_context,
            "representation_metadata": {
                "statistic": {"feature_groups": args.feature_groups},
            },
        },
    ).memory_bank

    hybrid = WeightedHybridRetriever(config={"fusion_method": "normalized_weighted_sum"})
    hybrid.add_retriever("ts", RawSeriesRetriever(config={"distance": args.ts_distance}))
    hybrid.add_retriever("text", TextSummaryRetriever(config={"method": args.text_method}))
    hybrid.add_retriever(
        "stat",
        StatKNNRetriever(config={"distance": args.stat_distance, "normalize": args.stat_normalize}),
    )
    hybrid.set_retriever_weight("ts", args.ts_weight)
    hybrid.set_retriever_weight("text", args.text_weight)
    hybrid.set_retriever_weight("stat", args.stat_weight)

    task_spec = TaskSpec(task_type=TaskType.CLASSIFICATION)
    true_labels = []
    predicted_labels = []
    per_query = []
    for sample in bundle.test.samples:
        query = QueryInstance(
            query_id=f"query__{sample.sample_id}",
            sample=sample,
            task_spec=task_spec,
            metadata=compute_query_metadata(sample, args.channel_id, args.feature_groups, llm_context),
        )
        retrieved = hybrid.retrieve(query=query, memory_bank=memory, top_k=args.k)
        labels = [ex.label for ex in retrieved.examples]
        true_labels.append(sample.y)
        predicted_labels.append(labels)
        per_query.append(
            {
                "sample_id": sample.sample_id,
                "true_label": sample.y,
                "retrieved_labels": labels,
                "retrieved_sample_ids": [ex.sample_id for ex in retrieved.examples],
            }
        )

    metrics = compute_topk_accuracy_and_precision_at_k(true_labels, predicted_labels, k=args.k)
    payload = {
        "dataset": bundle.dataset_name,
        "channel_id": args.channel_id,
        "top_k": args.k,
        "weights": {"ts": args.ts_weight, "text": args.text_weight, "stat": args.stat_weight},
        "metrics": metrics,
        "num_train": len(bundle.train.samples),
        "num_query": len(bundle.test.samples),
        "per_query": per_query,
    }
    if args.save_json:
        path = save_json(args.save_json, payload)
        print(f"Saved results to: {path}")
    else:
        print_json(payload)


if __name__ == "__main__":
    main()
