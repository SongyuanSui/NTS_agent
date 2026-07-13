#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from importlib import import_module
from typing import Any
from tqdm import tqdm

from common import (
    add_dataset_args,
    load_config_stack,
    load_dataset_from_config,
    print_json,
    resolve_config,
    save_json,
)
from core.enums import TaskType
from core.factories import build_agent, build_llm_client, build_pipeline, build_task
from memory.artifacts import get_memory_bank_path
from memory.memory_store import load_memory_bank_jsonl
from pipelines.memory_build_pipeline import MemoryBuildPipeline
from representations.statistics import compute_statistics_for_sample


def _infer_channel_ids_from_sample(sample: Any) -> list[int]:
    """Infer channel ids from a sample when pipeline config doesn't pin channels."""
    channels = getattr(sample, "channels", None)
    if channels is not None:
        try:
            n = len(channels)
            if n > 0:
                return list(range(int(n)))
        except Exception:
            pass

    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict) and "num_channels" in metadata:
        try:
            n = int(metadata["num_channels"])
            if n > 0:
                return list(range(n))
        except Exception:
            pass

    x = getattr(sample, "x", None)
    if x is None:
        return [0]
    try:
        ndim = int(getattr(x, "ndim", 1))
        if ndim <= 1:
            return [0]
        # Heuristic fallback when explicit channel metadata is unavailable.
        # For common TS layouts, channels are often the smaller axis.
        shape = tuple(int(s) for s in x.shape)
        num_channels = min(shape[0], shape[1]) if len(shape) >= 2 else int(shape[0])
        if num_channels <= 0:
            return [0]
        return list(range(num_channels))
    except Exception:
        return [0]


def _infer_label_space_from_samples(samples: list[Any]) -> list[str]:
    """Infer a stable label space from labeled samples, preserving first-seen order."""
    labels: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        label = getattr(sample, "y", None)
        if label is None:
            continue
        label_text = str(label)
        if label_text not in seen:
            seen.add(label_text)
            labels.append(label_text)
    return labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full end-to-end inference pipeline.")
    add_dataset_args(parser)
    parser.add_argument("--pipeline-config", type=str, default="configs/pipelines/end2end_multivariate.yaml")
    parser.add_argument("--task-config", type=str, default=None)
    parser.add_argument("--memory-config", type=str, default=None)
    parser.add_argument("--channel-ids", nargs="*", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--skip-memory-build", action="store_true")
    parser.add_argument(
        "--memory-bank-path",
        type=str,
        default=None,
        help=(
            "Path to a persisted memory_bank.jsonl to reuse with --skip-memory-build. "
            "If omitted, the path is derived from the memory config "
            "(outputs_root/memory/{dataset}_{experiment_name}/memory_bank.jsonl)."
        ),
    )
    parser.add_argument(
        "--retrieval-agent-override",
        type=str,
        default=None,
        choices=[
            "retrieval_agent_stat",
            "retrieval_agent_hybrid",
            "retrieval_agent_text",
            "retrieval_agent_ts",
        ],
        help="Override retrieval agent configured in pipeline components.",
    )
    parser.add_argument("--save-json", type=str, default=None)
    return parser


def _import_for_registration(pipeline_cfg: dict[str, Any], task_name: str, component_names: list[str]) -> None:
    # Ensure decorators register all runtime classes used by factories.
    import_module(f"tasks.{task_name}")
    for comp_name in component_names:
        if comp_name == task_name:
            continue
        # component names in config map to agents modules by convention.
        try:
            import_module(f"agents.{comp_name}")
        except ModuleNotFoundError:
            # Some component names may not be agents.
            pass
    pipeline_name = pipeline_cfg.get("pipeline", {}).get("name", "inference_pipeline")
    if pipeline_name:
        # name "inference_pipeline" maps to module pipelines.inference_pipeline
        import_module(f"pipelines.{pipeline_name}")


def _load_task_config(task_name: str, task_config_path: str | None) -> dict[str, Any]:
    path = task_config_path or f"configs/tasks/{task_name}.yaml"
    cfg = load_config_stack(path)
    if "task" not in cfg:
        raise ValueError(f"Task config '{path}' must contain top-level 'task'.")
    return cfg["task"]


def _load_agent_config(agent_name: str) -> dict[str, Any]:
    path = f"configs/agents/{agent_name}.yaml"
    cfg = load_config_stack(path)
    if "agent" not in cfg:
        raise ValueError(f"Agent config '{path}' must contain top-level 'agent'.")
    return cfg["agent"]


def _resolve_persisted_memory_bank_path(
    bundle: Any,
    mem_params: dict[str, Any],
    explicit_path: str | None,
) -> str:
    """Locate the persisted memory_bank.jsonl to reuse when skipping the build."""
    if explicit_path:
        return explicit_path

    outputs_root = mem_params.get("outputs_root", "outputs")
    experiment_name = mem_params.get("experiment_name", "memory_build")
    dataset_name = mem_params.get("dataset_name") or bundle.dataset_name
    run_dir = f"{outputs_root}/memory/{dataset_name}_{experiment_name}"
    return str(get_memory_bank_path(run_dir))


def _build_memory_bank(
    bundle: Any,
    task_type: TaskType,
    pipeline_cfg: dict[str, Any],
    memory_config_path: str | None,
    channel_ids_override: list[int] | None,
    skip_memory_build: bool,
    memory_bank_path: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    memory_cfg_path = memory_config_path or pipeline_cfg.get("memory", {}).get("memory_config")
    memory_cfg = load_config_stack(memory_cfg_path) if memory_cfg_path else {}

    mem_params = dict(memory_cfg.get("pipeline", {}).get("params", {}))
    should_build = bool(pipeline_cfg.get("memory", {}).get("build_before_inference", False))
    if skip_memory_build:
        should_build = False

    if skip_memory_build:
        # Reuse a previously persisted memory bank instead of rebuilding.
        bank_path = _resolve_persisted_memory_bank_path(bundle, mem_params, memory_bank_path)
        if not os.path.exists(bank_path):
            raise FileNotFoundError(
                f"--skip-memory-build set but no persisted memory bank at '{bank_path}'. "
                "Build it once (without --skip-memory-build) or pass --memory-bank-path."
            )
        print(f"[inference] reusing persisted memory bank: {bank_path}", flush=True)
        memory_bank = load_memory_bank_jsonl(bank_path)
        return memory_bank, {
            "memory_built": False,
            "memory_reused": True,
            "memory_bank_path": bank_path,
            "num_memory_entries": len(memory_bank.get_all()),
        }

    if not should_build:
        return None, {"memory_built": False}

    channel_ids = channel_ids_override
    if channel_ids is None:
        cfg_ids = mem_params.get("selected_channel_ids")
        channel_ids = [int(x) for x in cfg_ids] if cfg_ids else None

    from representations.raw_series import RawSeriesRepresentation
    from representations.statistics import StatisticsRepresentation
    from representations.text_summary import TextSummaryRepresentation

    summary_style = (
        mem_params.get("representation_metadata", {})
        .get("summary", {})
        .get("style", "statistical")
    )
    feature_groups = (
        mem_params.get("representation_metadata", {})
        .get("statistic", {})
        .get("feature_groups", None)
    )

    memory_pipeline = MemoryBuildPipeline(
        components={
            "ts_representation": RawSeriesRepresentation(),
            "summary_representation": TextSummaryRepresentation(config={"style": summary_style}),
            "statistic_representation": StatisticsRepresentation(),
        },
        config={"task_type": task_type.value},
    )

    context = {
        "persist_memory": bool(mem_params.get("persist_memory", False)),
        "dataset_name": bundle.dataset_name,
        "outputs_root": mem_params.get("outputs_root", "outputs"),
        "experiment_name": mem_params.get("experiment_name", "inference_memory_build"),
        "representation_metadata": {
            "summary": {"style": summary_style},
            "statistic": {"feature_groups": feature_groups},
        },
    }

    llm_cfg = (
        pipeline_cfg.get("pipeline", {})
        .get("params", {})
        .get("llm")
    )
    if isinstance(llm_cfg, dict) and llm_cfg:
        context["llm_client"] = build_llm_client(llm_cfg)

    result = memory_pipeline.build_memory_bank(
        samples=bundle.train.samples,
        task_type=task_type,
        channel_ids=channel_ids,
        context=context,
    )
    meta = {"memory_built": True, "memory_metadata": result.metadata, "memory_artifacts": result.artifacts}
    return result.memory_bank, meta


def main() -> None:
    args = build_parser().parse_args()

    print("[inference] loading configs", flush=True)

    pipeline_root_cfg = load_config_stack(args.pipeline_config)
    if "pipeline" not in pipeline_root_cfg:
        raise ValueError("Pipeline config must contain top-level 'pipeline'.")

    components_cfg = dict(pipeline_root_cfg.get("components", {}))
    if not components_cfg:
        raise ValueError("Pipeline config must contain top-level 'components'.")

    if args.retrieval_agent_override:
        components_cfg["retrieval_agent"] = args.retrieval_agent_override

    task_name = str(components_cfg.get("task", "classification"))
    _import_for_registration(pipeline_root_cfg, task_name, list(components_cfg.values()))

    print("[inference] loading task config", flush=True)
    task_cfg = _load_task_config(task_name=task_name, task_config_path=args.task_config)

    print("[inference] loading dataset", flush=True)
    data_cfg = resolve_config(args.config, "configs/data/classification.yaml")
    bundle = load_dataset_from_config(
        data_cfg,
        dataset_loader=args.dataset_loader,
        dataset_name=args.dataset,
        base_dir=args.base_dir,
        max_samples_per_split=args.max_samples_per_split,
    )

    task_type = TaskType(task_cfg["task_spec"]["task_type"])

    if task_type == TaskType.CLASSIFICATION:
        task_spec_cfg = task_cfg.setdefault("task_spec", {})
        if not task_spec_cfg.get("label_space"):
            task_spec_cfg["label_space"] = _infer_label_space_from_samples(bundle.train.samples)

    print("[inference] building pipeline components", flush=True)
    components: dict[str, Any] = {
        "task": build_task(task_cfg),
    }
    for key, name in components_cfg.items():
        if key == "task":
            continue
        # InferencePipeline requires this subset; include optional ones if present.
        agent_cfg = _load_agent_config(str(name))
        components[key] = build_agent(agent_cfg)

    pipeline = build_pipeline(pipeline_root_cfg["pipeline"], components=components)

    print("[inference] building memory bank", flush=True)
    memory_bank, memory_meta = _build_memory_bank(
        bundle=bundle,
        task_type=task_type,
        pipeline_cfg=pipeline_root_cfg,
        memory_config_path=args.memory_config,
        channel_ids_override=args.channel_ids,
        skip_memory_build=bool(args.skip_memory_build),
        memory_bank_path=args.memory_bank_path,
    )
    print("[inference] memory bank ready", flush=True)

    samples = list(bundle.test.samples)
    if args.max_test_samples is not None:
        samples = samples[: max(0, int(args.max_test_samples))]

    selected_channel_ids = args.channel_ids
    if selected_channel_ids is None:
        selected_channel_ids = pipeline.get_config("selected_channel_ids")

    records: list[dict[str, Any]] = []
    print(f"[inference] running {len(samples)} test samples", flush=True)
    for sample in tqdm(samples, desc="Inference", unit="sample"):
        run_context: dict[str, Any] = {
            "dataset_name": bundle.dataset_name,
        }
        if memory_bank is not None:
            run_context["memory_bank"] = memory_bank
        if selected_channel_ids is not None:
            run_context["selected_channel_ids"] = selected_channel_ids
            # Provide statistic query features for retrievers that require it.
            if len(selected_channel_ids) == 1:
                run_context["query_stat_dict"] = compute_statistics_for_sample(
                    sample,
                    channel_id=int(selected_channel_ids[0]),
                    feature_groups=None,
                )
            elif len(selected_channel_ids) > 1:
                run_context["query_stat_by_channel"] = {
                    int(cid): compute_statistics_for_sample(
                        sample,
                        channel_id=int(cid),
                        feature_groups=None,
                    )
                    for cid in selected_channel_ids
                }
        else:
            # If channels are selected dynamically by the decomposer, precompute for all channels.
            inferred_channel_ids = _infer_channel_ids_from_sample(sample)
            run_context["query_stat_by_channel"] = {
                int(cid): compute_statistics_for_sample(
                    sample,
                    channel_id=int(cid),
                    feature_groups=None,
                )
                for cid in inferred_channel_ids
            }

        result = pipeline.run(sample, context=run_context)
        pred = result.prediction
        reasoner_output = result.intermediates.get("reasoner_output")

        channel_decisions = []
        if reasoner_output is not None and hasattr(reasoner_output, "channel_decisions"):
            for decision in reasoner_output.channel_decisions:
                channel_decisions.append(
                    {
                        "channel_id": decision.channel_id,
                        "prediction": decision.prediction,
                        "confidence": decision.confidence,
                        "metadata": decision.metadata,
                    }
                )

        records.append(
            {
                "sample_id": sample.sample_id,
                "true_label": sample.y,
                "prediction": pred.prediction,
                "confidence": pred.confidence,
                "reasoning": pred.reasoning,
                "prediction_metadata": pred.metadata,
                "channel_decisions": channel_decisions,
            }
        )

    payload = {
        "dataset": bundle.dataset_name,
        "num_train": len(bundle.train.samples),
        "num_test": len(bundle.test.samples),
        "num_run": len(records),
        "pipeline": pipeline.name,
        "pipeline_config": args.pipeline_config,
        "memory": memory_meta,
        "records": records,
    }

    if args.save_json:
        path = save_json(args.save_json, payload)
        print(f"Saved inference results to: {path}")
    else:
        print_json(payload)


if __name__ == "__main__":
    main()
