from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from core.constants import (
    DEFAULT_MEMORY_BANK_FILENAME,
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_SELECTED_CHANNELS_FILENAME,
    DEFAULT_STAT_INDEX_FILENAME,
    DEFAULT_TOP_K,
)
from core.enums import TaskType
from core.registry import PIPELINE_REGISTRY
from core.schemas import PipelineResult, PredictionRecord, QueryInstance, TaskSpec, TimeSeriesSample
from evaluation.retrieval_metrics import compute_topk_accuracy_and_precision_at_k
from memory.artifacts import (
    ensure_run_dir,
    get_build_meta_path,
    get_index_stat_path,
    get_memory_run_dir,
    get_memory_bank_path,
    save_build_meta,
)
from memory.indexing import build_stat_index, save_stat_index
from memory.memory_bank import MemoryBank
from memory.memory_store import load_memory_bank_jsonl, save_memory_bank_jsonl
from memory.schemas import MemoryEntry
from pipelines.pipeline_base import BasePipeline
from representations.schemas import RepresentationInput
from utils.io import read_json, write_json


@dataclass
class MemoryPersistenceConfig:
    """Configuration for memory persistence and reuse strategies."""

    persist_memory: bool = False
    reuse_memory: bool = True
    force_rebuild: bool = False
    dataset_name: Optional[str] = None
    experiment_name: Optional[str] = None
    outputs_root: str = DEFAULT_OUTPUTS_ROOT
    selected_channel_ids: Optional[list[int]] = None

    def validate(self) -> None:
        """Validate configuration when persistence is enabled."""
        if self.persist_memory and (not self.dataset_name or not self.experiment_name):
            raise ValueError(
                "dataset_name and experiment_name are required when persist_memory=True."
            )


@PIPELINE_REGISTRY.decorator("stat_feature_retrieval_pipeline")
class StatFeatureRetrievalPipeline(BasePipeline):
    """
    Lightweight pipeline for stat-feature retrieval experiments.

    Scope
    -----
    - Compute statistic representation for query/train samples
    - Build an in-memory candidate bank (no artifact persistence)
    - Run retriever and report retrieval metrics

    This pipeline is intended for quick E2E validation of feature effectiveness,
    similar to script-style retrieval experiments.
    """

    REQUIRED_COMPONENTS = (
        "representation",
        "retriever",
    )

    def validate_components(self) -> None:
        super().validate_components()

        for key in self.REQUIRED_COMPONENTS:
            if not self.has_component(key):
                raise KeyError(f"{self.name}: required component '{key}' is missing.")

    def _run_impl(
        self,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Run retrieval for a single query sample.

        Required context keys
        ---------------------
        - memory_bank: MemoryBank or iterable candidates

        Optional context keys
        ---------------------
        - top_k: override top-k
        - task_type: TaskType or valid string
        - query_label: override sample.y for metric/e2e debugging
        """
        context = self.normalize_context(context)

        memory_bank = context.get("memory_bank")
        if memory_bank is None:
            raise ValueError(
                f"{self.name}: context['memory_bank'] is required for single-sample retrieval."
            )

        top_k = int(context.get("top_k", self.get_config("top_k", DEFAULT_TOP_K)))
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        task_type = self._resolve_task_type(context)
        channel_id = int(self.get_config("channel_id", 0))

        query, stat_payload = self._build_query_with_stat(
            sample=sample,
            task_type=task_type,
            channel_id=channel_id,
        )

        retriever = self.require_component("retriever")
        retrieved_set = retriever.retrieve(
            query=query,
            memory_bank=memory_bank,
            top_k=top_k,
            context={"query_stat_dict": stat_payload},
        )

        top1_label = None
        confidence = None
        if len(retrieved_set.examples) > 0:
            top_example = retrieved_set.examples[0]
            top1_label = top_example.label
            if top_example.score is not None:
                # Distance-based score by default for stat retriever.
                confidence = 1.0 / (1.0 + float(top_example.score.value))

        prediction = PredictionRecord(
            sample_id=sample.sample_id,
            task_type=task_type,
            prediction=top1_label,
            confidence=confidence,
            metadata={
                "top_k": top_k,
                "num_retrieved": len(retrieved_set.examples),
            },
        )

        return PipelineResult(
            prediction=prediction,
            intermediates={
                "query": query,
                "retrieved_set": retrieved_set,
            },
            metadata={
                "pipeline_type": "stat_feature_retrieval",
            },
        )

    def build_memory_bank(
        self,
        samples: Iterable[TimeSeriesSample],
        task_type: TaskType | str,
        channel_id: int = 0,
    ) -> MemoryBank:
        """
        Build in-memory stat candidates from samples.

        This function intentionally does not persist artifacts.
        """
        sample_list = self.validate_samples(samples)
        resolved_task_type = self._coerce_task_type(task_type)

        stat_by_sample_id = self._compute_stat_payloads(
            samples=sample_list,
            channel_id=channel_id,
        )

        entries: list[MemoryEntry] = []
        for sample in sample_list:
            if sample.sample_id not in stat_by_sample_id:
                raise ValueError(
                    f"Missing statistic representation for sample_id='{sample.sample_id}'"
                )

            entries.append(
                MemoryEntry(
                    entry_id=f"{sample.sample_id}__ch{int(channel_id)}",
                    sample_id=sample.sample_id,
                    channel_id=int(channel_id),
                    task_type=resolved_task_type,
                    label=sample.y,
                    statistic_view=stat_by_sample_id[sample.sample_id],
                    metadata={
                        "dataset_name": sample.metadata.get("dataset_name"),
                        "split": sample.metadata.get("split"),
                    },
                )
            )

        return MemoryBank(entries=entries)

    def evaluate_split(
        self,
        train_samples: Iterable[TimeSeriesSample],
        query_samples: Iterable[TimeSeriesSample],
        task_type: TaskType | str,
        top_k: Optional[int] = None,
        channel_id: int = 0,
        memory_bank: Optional[MemoryBank] = None,
        memory_config: Optional[MemoryPersistenceConfig] = None,
    ) -> dict[str, Any]:
        """
        Build memory from train split and evaluate retrieval on query split.

        Parameters
        ----------
        train_samples : Iterable[TimeSeriesSample]
            Training samples for memory bank.
        query_samples : Iterable[TimeSeriesSample]
            Query samples for evaluation.
        task_type : TaskType | str
            Task type (classification, prediction, etc.).
        top_k : Optional[int]
            Number of top candidates to retrieve. Defaults to pipeline config.
        channel_id : int
            Channel ID for univariate extraction. Defaults to 0.
        memory_bank : Optional[MemoryBank]
            Pre-built memory bank (bypasses memory prep logic).
        memory_config : Optional[MemoryPersistenceConfig]
            Memory persistence configuration. If None, creates default (no persistence).

        Returns
        -------
        dict containing aggregate retrieval metrics and per-query labels.
        """
        train_list = self.validate_samples(train_samples)
        query_list = self.validate_samples(query_samples)

        k = int(top_k if top_k is not None else self.get_config("top_k", DEFAULT_TOP_K))
        if k <= 0:
            raise ValueError("top_k must be a positive integer.")

        resolved_task_type = self._coerce_task_type(task_type)
        resolved_channel_id = int(channel_id)

        # Default to no persistence if not specified
        if memory_config is None:
            memory_config = MemoryPersistenceConfig()
        memory_config.validate()

        memory_bank, memory_artifacts = self._prepare_memory_for_evaluation(
            train_samples=train_list,
            task_type=resolved_task_type,
            channel_id=resolved_channel_id,
            memory_bank=memory_bank,
            config=memory_config,
        )

        true_labels: list[Any] = []
        predicted_labels: list[list[Any]] = []

        for sample in query_list:
            result = self.run(
                sample=sample,
                context={
                    "memory_bank": memory_bank,
                    "top_k": k,
                    "task_type": resolved_task_type,
                },
            )
            retrieved_set = result.intermediates["retrieved_set"]
            true_labels.append(sample.y)
            predicted_labels.append([ex.label for ex in retrieved_set.examples])

        metrics = compute_topk_accuracy_and_precision_at_k(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            k=k,
        )

        return {
            "metrics": metrics,
            "num_train": len(train_list),
            "num_query": len(query_list),
            "top_k": k,
            "channel_id": resolved_channel_id,
            "task_type": resolved_task_type.value,
            **({"memory_artifacts": memory_artifacts} if memory_artifacts is not None else {}),
        }

    def _prepare_memory_for_evaluation(
        self,
        train_samples: list[TimeSeriesSample],
        task_type: TaskType,
        channel_id: int,
        memory_bank: Optional[MemoryBank],
        config: MemoryPersistenceConfig,
    ) -> tuple[MemoryBank, Optional[dict[str, str]]]:
        """Prepare memory bank for evaluation with optional artifact persistence/reuse."""
        if memory_bank is not None:
            return memory_bank, None

        if not config.persist_memory:
            return (
                self.build_memory_bank(
                    samples=train_samples,
                    task_type=task_type,
                    channel_id=channel_id,
                ),
                None,
            )

        if config.reuse_memory:
            return self.load_or_build_memory(
                samples=train_samples,
                dataset_name=config.dataset_name,
                experiment_name=config.experiment_name,
                task_type=task_type,
                channel_id=channel_id,
                outputs_root=config.outputs_root,
                selected_channel_ids=config.selected_channel_ids,
                force_rebuild=config.force_rebuild,
            )

        built_memory = self.build_memory_bank(
            samples=train_samples,
            task_type=task_type,
            channel_id=channel_id,
        )
        artifact_info = self.persist_memory_artifacts(
            memory_bank=built_memory,
            dataset_name=config.dataset_name,
            experiment_name=config.experiment_name,
            task_type=task_type,
            outputs_root=config.outputs_root,
            selected_channel_ids=config.selected_channel_ids,
            extra_meta={
                "task_type": task_type.value,
                "channel_id": channel_id,
                "representation_metadata": dict(self.get_config("representation_metadata", {})),
                "num_train_samples": len(train_samples),
            },
        )
        artifact_info["memory_source"] = "built"
        return built_memory, artifact_info

    def persist_memory_artifacts(
        self,
        memory_bank: MemoryBank,
        dataset_name: str,
        experiment_name: str,
        task_type: TaskType | str,
        outputs_root: str = DEFAULT_OUTPUTS_ROOT,
        selected_channel_ids: Optional[list[int]] = None,
        extra_meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, str]:
        """
        Persist memory bank and stat index under outputs/memory/<dataset>_<experiment>/.
        """
        resolved_task_type = self._coerce_task_type(task_type)

        run_dir = ensure_run_dir(
            outputs_root=outputs_root,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
        )

        memory_path = get_memory_bank_path(run_dir, filename=DEFAULT_MEMORY_BANK_FILENAME)
        save_memory_bank_jsonl(memory_bank=memory_bank, path=memory_path)

        stat_index = build_stat_index(memory_bank)
        index_stat_path = get_index_stat_path(run_dir, filename=DEFAULT_STAT_INDEX_FILENAME)
        save_stat_index(stat_index, index_stat_path)

        if selected_channel_ids is None:
            selected_channel_ids = sorted({int(entry.channel_id) for entry in memory_bank.get_all()})

        selected_channels_path = run_dir / DEFAULT_SELECTED_CHANNELS_FILENAME
        write_json(
            selected_channels_path,
            {
                "dataset_name": dataset_name,
                "experiment_name": experiment_name,
                "task_type": resolved_task_type.value,
                "selected_channel_ids": [int(x) for x in selected_channel_ids],
            },
            indent=2,
            ensure_ascii=False,
        )

        build_meta_path = save_build_meta(
            run_dir=run_dir,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            task_type=resolved_task_type.value,
            dataset_path=dataset_name,
            extra_meta={
                "num_entries": len(memory_bank),
                "selected_channels_path": str(selected_channels_path),
                "memory_bank_path": str(memory_path),
                "index_stat_path": str(index_stat_path),
                **(extra_meta or {}),
            },
        )

        return {
            "run_dir": str(run_dir),
            "selected_channels_path": str(selected_channels_path),
            "memory_bank_path": str(memory_path),
            "index_stat_path": str(index_stat_path),
            "build_meta_path": str(build_meta_path),
        }

    def load_or_build_memory(
        self,
        samples: Iterable[TimeSeriesSample],
        dataset_name: str,
        experiment_name: str,
        task_type: TaskType | str,
        channel_id: int = 0,
        outputs_root: str = DEFAULT_OUTPUTS_ROOT,
        selected_channel_ids: Optional[list[int]] = None,
        force_rebuild: bool = False,
    ) -> tuple[MemoryBank, dict[str, str]]:
        """
        Load persisted memory artifacts when compatible, otherwise rebuild and persist.
        """
        sample_list = self.validate_samples(samples)
        resolved_task_type = self._coerce_task_type(task_type)
        resolved_channel_id = int(channel_id)
        rep_meta = dict(self.get_config("representation_metadata", {}))

        run_dir = get_memory_run_dir(
            outputs_root=outputs_root,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            create=False,
        )
        memory_path = get_memory_bank_path(run_dir, filename=DEFAULT_MEMORY_BANK_FILENAME)
        index_stat_path = get_index_stat_path(run_dir, filename=DEFAULT_STAT_INDEX_FILENAME)
        build_meta_path = get_build_meta_path(run_dir)

        expected_meta = {
            "task_type": resolved_task_type.value,
            "channel_id": resolved_channel_id,
            "representation_metadata": rep_meta,
            "num_train_samples": len(sample_list),
        }

        can_try_load = (
            not force_rebuild
            and run_dir.exists()
            and memory_path.exists()
            and index_stat_path.exists()
            and build_meta_path.exists()
        )

        if can_try_load:
            try:
                build_meta = read_json(build_meta_path)
                if self._is_memory_meta_compatible(build_meta=build_meta, expected_meta=expected_meta):
                    loaded_memory = load_memory_bank_jsonl(memory_path)
                    return loaded_memory, {
                        "memory_source": "loaded",
                        "run_dir": str(run_dir),
                        "selected_channels_path": str(run_dir / DEFAULT_SELECTED_CHANNELS_FILENAME),
                        "memory_bank_path": str(memory_path),
                        "index_stat_path": str(index_stat_path),
                        "build_meta_path": str(build_meta_path),
                    }
            except Exception:
                # If loading/parsing fails, fall back to rebuild to keep run robust.
                pass

        built_memory = self.build_memory_bank(
            samples=sample_list,
            task_type=resolved_task_type,
            channel_id=resolved_channel_id,
        )
        artifact_info = self.persist_memory_artifacts(
            memory_bank=built_memory,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            task_type=resolved_task_type,
            outputs_root=outputs_root,
            selected_channel_ids=selected_channel_ids,
            extra_meta=expected_meta,
        )
        artifact_info["memory_source"] = "built"
        return built_memory, artifact_info

    def _is_memory_meta_compatible(
        self,
        build_meta: Any,
        expected_meta: dict[str, Any],
    ) -> bool:
        if not isinstance(build_meta, dict):
            return False

        for key, expected_value in expected_meta.items():
            if key not in build_meta:
                return False
            if build_meta[key] != expected_value:
                return False

        memory_bank_path = build_meta.get("memory_bank_path")
        if memory_bank_path and not Path(memory_bank_path).exists():
            return False

        index_stat_path = build_meta.get("index_stat_path")
        if index_stat_path and not Path(index_stat_path).exists():
            return False

        return True

    def _build_query_with_stat(
        self,
        sample: TimeSeriesSample,
        task_type: TaskType,
        channel_id: int,
    ) -> tuple[QueryInstance, dict[str, float]]:
        stat_by_sample_id = self._compute_stat_payloads(samples=[sample], channel_id=channel_id)
        stat_payload = stat_by_sample_id.get(sample.sample_id)
        if stat_payload is None:
            raise ValueError(f"Failed to compute statistic_view for sample_id='{sample.sample_id}'")

        query = QueryInstance(
            query_id=f"query__{sample.sample_id}",
            sample=sample,
            task_spec=TaskSpec(task_type=task_type, label_space=[]),
            metadata={"statistic_view": stat_payload},
        )
        return query, stat_payload

    def _compute_stat_payloads(
        self,
        samples: list[TimeSeriesSample],
        channel_id: int,
    ) -> dict[str, dict[str, float]]:
        representation = self.require_component("representation")

        rep_input = RepresentationInput(
            samples=samples,
            channel_id=int(channel_id),
            metadata=dict(self.get_config("representation_metadata", {})),
        )
        rep_output = representation.run(rep_input)

        stat_by_sample_id: dict[str, dict[str, float]] = {}
        input_sample_ids = {sample.sample_id for sample in samples}

        # Preferred mapping: explicit sample_id in record metadata.
        for record in rep_output.records:
            sample_id = str(record.metadata.get("sample_id", "")).strip()
            if sample_id and sample_id in input_sample_ids:
                stat_by_sample_id[sample_id] = dict(record.payload)

        has_aggregate_payload = any(
            isinstance(record.payload, dict) and "statistical_features" in record.payload
            for record in rep_output.records
        )

        # Fallback mapping: positional alignment with input samples.
        if not has_aggregate_payload and len(rep_output.records) == len(samples):
            for sample, record in zip(samples, rep_output.records):
                if sample.sample_id not in stat_by_sample_id:
                    stat_by_sample_id[sample.sample_id] = dict(record.payload)


        return stat_by_sample_id

    def _resolve_task_type(self, context: dict[str, Any]) -> TaskType:
        task_type = context.get("task_type", self.get_config("task_type", TaskType.CLASSIFICATION))
        return self._coerce_task_type(task_type)

    def _coerce_task_type(self, task_type: TaskType | str) -> TaskType:
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        if not isinstance(task_type, TaskType):
            raise TypeError("task_type must be TaskType or a valid task_type string.")
        return task_type
