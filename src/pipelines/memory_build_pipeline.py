from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from tqdm import tqdm

from core.constants import (
    DEFAULT_MEMORY_BANK_FILENAME,
    DEFAULT_OUTPUTS_ROOT,
    DEFAULT_SELECTED_CHANNELS_FILENAME,
    DEFAULT_STAT_INDEX_FILENAME,
)
from core.enums import RepresentationType, TaskType
from core.registry import PIPELINE_REGISTRY
from core.schemas import BatchPredictionRecord, PipelineResult, PredictionRecord, TimeSeriesSample
from memory.artifacts import ensure_run_dir, get_index_stat_path, get_memory_bank_path, save_build_meta
from memory.indexing import build_stat_index, save_stat_index
from memory.memory_bank import MemoryBank
from memory.memory_store import save_memory_bank_jsonl
from memory.schemas import MemoryEntry
from pipelines.pipeline_base import BasePipeline
from representations.schemas import RepresentationInput, RepresentationOutput
from utils.io import write_json


@dataclass(slots=True)
class MemoryBuildResult:
    """Structured result returned by MemoryBuildPipeline.build_memory_bank."""

    memory_bank: MemoryBank
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@PIPELINE_REGISTRY.decorator("memory_build_pipeline")
class MemoryBuildPipeline(BasePipeline):
    """
    Pipeline for building retrieval memory entries from time-series samples.

    Components may be supplied with any of these keys:
    - ts_representation
    - summary_representation
    - statistic_representation
    - representation (used as a fallback for statistic_representation)
    """

    REPRESENTATION_KEYS = {
        RepresentationType.TS: "ts_representation",
        RepresentationType.SUMMARY: "summary_representation",
        RepresentationType.STATISTIC: "statistic_representation",
    }

    def validate_components(self) -> None:
        super().validate_components()
        if not any(
            key in self.components
            for key in (
                "ts_representation",
                "summary_representation",
                "statistic_representation",
                "representation",
            )
        ):
            raise KeyError(
                f"{self.name}: at least one representation component is required."
            )

    def _run_impl(
        self,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> PipelineResult:
        context = self.normalize_context(context)
        task_type = self._resolve_task_type(context)
        channel_ids = self._resolve_channel_ids(samples=[sample], context=context)
        result = self.build_memory_bank(
            samples=[sample],
            task_type=task_type,
            channel_ids=channel_ids,
            context=context,
        )

        prediction = PredictionRecord(
            sample_id=sample.sample_id,
            task_type=task_type,
            prediction=len(result.memory_bank),
            confidence=None,
            metadata={
                "pipeline_name": self.name,
                "num_memory_entries": len(result.memory_bank),
                "channel_ids": channel_ids,
            },
        )
        return PipelineResult(
            prediction=prediction,
            intermediates={"memory_bank": result.memory_bank},
            metadata={**result.metadata, "artifacts": result.artifacts},
        )

    def run_batch(
        self,
        samples: Iterable[TimeSeriesSample],
        context: Optional[dict[str, Any]] = None,
    ) -> BatchPredictionRecord:
        context = self.normalize_context(context)
        sample_list = self.validate_samples(samples)
        task_type = self._resolve_task_type(context)
        channel_ids = self._resolve_channel_ids(samples=sample_list, context=context)
        result = self.build_memory_bank(
            samples=sample_list,
            task_type=task_type,
            channel_ids=channel_ids,
            context=context,
        )

        records = [
            PredictionRecord(
                sample_id=sample.sample_id,
                task_type=task_type,
                prediction=len(result.memory_bank.get_by_sample_id(sample.sample_id)),
                metadata={
                    "pipeline_name": self.name,
                    "num_memory_entries_for_sample": len(
                        result.memory_bank.get_by_sample_id(sample.sample_id)
                    ),
                },
            )
            for sample in sample_list
        ]
        return BatchPredictionRecord(
            records=records,
            metadata={
                "pipeline_name": self.name,
                "num_samples": len(sample_list),
                "num_memory_entries": len(result.memory_bank),
                "channel_ids": channel_ids,
                "artifacts": result.artifacts,
                "memory_bank": result.memory_bank,
            },
        )

    def build_memory_bank(
        self,
        samples: Iterable[TimeSeriesSample],
        task_type: TaskType | str,
        channel_ids: Optional[Iterable[int]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> MemoryBuildResult:
        context = self.normalize_context(context)
        sample_list = self.validate_samples(samples)
        resolved_task_type = self._coerce_task_type(task_type)
        resolved_channel_ids = self._resolve_channel_ids(sample_list, context, channel_ids)

        self.log_info(
            context,
            "MemoryBuildPipeline '%s': start memory build for %d samples across %d channels",
            self.name,
            len(sample_list),
            len(resolved_channel_ids),
        )

        memory_bank = MemoryBank(logger=self.get_logger(context))
        for channel_id in tqdm(resolved_channel_ids, desc="Memory build", unit="channel"):
            self.log_info(
                context,
                "MemoryBuildPipeline '%s': building channel_id=%d",
                self.name,
                channel_id,
            )
            views_by_type = self._compute_views_for_channel(
                samples=sample_list,
                channel_id=channel_id,
                context=context,
            )
            for sample in sample_list:
                entry = self._build_entry(
                    sample=sample,
                    task_type=resolved_task_type,
                    channel_id=channel_id,
                    views_by_type=views_by_type,
                    shared_metadata=dict(context.get("entry_metadata", {})),
                )
                memory_bank.add(entry)

            self.log_info(
                context,
                "MemoryBuildPipeline '%s': finished channel_id=%d with %d entries",
                self.name,
                channel_id,
                len(sample_list),
            )

        artifacts: dict[str, str] = {}
        if bool(context.get("persist_memory", self.get_config("persist_memory", False))):
            self.log_info(
                context,
                "MemoryBuildPipeline '%s': persisting memory artifacts",
                self.name,
            )
            artifacts = self.persist_memory_artifacts(
                memory_bank=memory_bank,
                task_type=resolved_task_type,
                channel_ids=resolved_channel_ids,
                context=context,
            )

        self.log_info(
            context,
            "MemoryBuildPipeline '%s': finished memory build with %d memory entries",
            self.name,
            len(memory_bank),
        )

        return MemoryBuildResult(
            memory_bank=memory_bank,
            artifacts=artifacts,
            metadata={
                "pipeline_name": self.name,
                "task_type": resolved_task_type.value,
                "num_samples": len(sample_list),
                "num_memory_entries": len(memory_bank),
                "channel_ids": resolved_channel_ids,
                "memory_summary": memory_bank.summary(log=False),
            },
        )

    def persist_memory_artifacts(
        self,
        memory_bank: MemoryBank,
        task_type: TaskType | str,
        channel_ids: list[int],
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, str]:
        context = self.normalize_context(context)
        resolved_task_type = self._coerce_task_type(task_type)
        dataset_name = context.get("dataset_name", self.get_config("dataset_name"))
        experiment_name = context.get("experiment_name", self.get_config("experiment_name"))
        if not dataset_name or not experiment_name:
            raise ValueError(
                f"{self.name}: dataset_name and experiment_name are required to persist memory."
            )

        outputs_root = context.get("outputs_root", self.get_config("outputs_root", DEFAULT_OUTPUTS_ROOT))
        run_dir = ensure_run_dir(
            outputs_root=outputs_root,
            dataset_name=str(dataset_name),
            experiment_name=str(experiment_name),
        )

        memory_path = get_memory_bank_path(run_dir, DEFAULT_MEMORY_BANK_FILENAME)
        save_memory_bank_jsonl(memory_bank, memory_path)

        artifacts = {
            "run_dir": str(run_dir),
            "memory_bank_path": str(memory_path),
        }

        if any(entry.statistic_view is not None for entry in memory_bank.get_all()):
            stat_index = build_stat_index(memory_bank)
            stat_index_path = get_index_stat_path(run_dir, DEFAULT_STAT_INDEX_FILENAME)
            save_stat_index(stat_index, stat_index_path)
            artifacts["index_stat_path"] = str(stat_index_path)

        selected_channels_path = Path(run_dir) / DEFAULT_SELECTED_CHANNELS_FILENAME
        write_json(
            selected_channels_path,
            {
                "dataset_name": str(dataset_name),
                "experiment_name": str(experiment_name),
                "task_type": resolved_task_type.value,
                "selected_channel_ids": [int(x) for x in channel_ids],
            },
            indent=2,
            ensure_ascii=False,
        )
        artifacts["selected_channels_path"] = str(selected_channels_path)

        build_meta_path = save_build_meta(
            run_dir=run_dir,
            dataset_name=str(dataset_name),
            experiment_name=str(experiment_name),
            task_type=resolved_task_type.value,
            dataset_path=str(context.get("dataset_path", dataset_name)),
            extra_meta={
                "pipeline_name": self.name,
                "num_entries": len(memory_bank),
                "selected_channel_ids": [int(x) for x in channel_ids],
                **artifacts,
            },
        )
        artifacts["build_meta_path"] = str(build_meta_path)
        return artifacts

    def _compute_views_for_channel(
        self,
        samples: list[TimeSeriesSample],
        channel_id: int,
        context: dict[str, Any],
    ) -> dict[RepresentationType, dict[str, Any]]:
        views_by_type: dict[RepresentationType, dict[str, Any]] = {}
        metadata_by_type = dict(context.get("representation_metadata", {}))

        for rep_type, key in self.REPRESENTATION_KEYS.items():
            component = self.get_component(key)
            if component is None and rep_type == RepresentationType.STATISTIC:
                component = self.get_component("representation")
            if component is None:
                continue

            rep_context = {**context, "stage": f"memory_build_{rep_type.value}"}
            rep_input = RepresentationInput(
                samples=samples,
                channel_id=channel_id,
                metadata=dict(metadata_by_type.get(rep_type.value, metadata_by_type)),
            )
            output = component.run(rep_input, context=rep_context)
            if not isinstance(output, RepresentationOutput):
                raise TypeError(
                    f"{self.name}: component '{key}' must return RepresentationOutput."
                )
            views_by_type[rep_type] = self._records_to_sample_payloads(output, samples)

        return views_by_type

    def _records_to_sample_payloads(
        self,
        output: RepresentationOutput,
        samples: list[TimeSeriesSample],
    ) -> dict[str, Any]:
        sample_ids = {sample.sample_id for sample in samples}
        payloads: dict[str, Any] = {}

        from utils.retrieval_compat import unwrap_payload

        for record in output.records:
            sample_id = str(record.metadata.get("sample_id", "")).strip()
            if sample_id in sample_ids:
                payloads[sample_id] = unwrap_payload(record.payload)

        if len(payloads) < len(samples) and output.num_records == len(samples):
            for sample, record in zip(samples, output.records):
                payloads.setdefault(sample.sample_id, unwrap_payload(record.payload))

        return payloads

    def _build_entry(
        self,
        sample: TimeSeriesSample,
        task_type: TaskType,
        channel_id: int,
        views_by_type: dict[RepresentationType, dict[str, Any]],
        shared_metadata: dict[str, Any],
    ) -> MemoryEntry:
        ts_view = views_by_type.get(RepresentationType.TS, {}).get(sample.sample_id)
        summary_view = views_by_type.get(RepresentationType.SUMMARY, {}).get(sample.sample_id)
        statistic_view = views_by_type.get(RepresentationType.STATISTIC, {}).get(sample.sample_id)

        if ts_view is None and summary_view is None and statistic_view is None:
            raise ValueError(
                f"{self.name}: no representation view produced for sample_id={sample.sample_id}, "
                f"channel_id={channel_id}."
            )

        metadata = {**sample.metadata, **shared_metadata}
        return MemoryEntry(
            entry_id=f"{sample.sample_id}__ch{int(channel_id)}",
            sample_id=sample.sample_id,
            channel_id=int(channel_id),
            task_type=task_type,
            label=sample.y,
            ts_view=ts_view,
            summary_view=summary_view,
            statistic_view=statistic_view,
            metadata=metadata,
        )

    def _resolve_channel_ids(
        self,
        samples: list[TimeSeriesSample],
        context: Optional[dict[str, Any]] = None,
        channel_ids: Optional[Iterable[int]] = None,
    ) -> list[int]:
        if channel_ids is None and context is not None:
            channel_ids = context.get("selected_channel_ids", context.get("channel_ids"))
        if channel_ids is None:
            channel_ids = self.get_config("selected_channel_ids", self.get_config("channel_ids", None))
        if channel_ids is None:
            max_channels = max(sample.num_channels for sample in samples)
            channel_ids = list(range(max_channels))

        resolved = sorted({int(channel_id) for channel_id in channel_ids})
        if not resolved:
            raise ValueError("channel_ids must contain at least one channel.")
        for channel_id in resolved:
            if channel_id < 0:
                raise ValueError("channel_ids must be non-negative.")
            for sample in samples:
                if channel_id >= sample.num_channels:
                    raise ValueError(
                        f"channel_id={channel_id} is out of range for sample "
                        f"'{sample.sample_id}' with {sample.num_channels} channel(s)."
                    )
        return resolved

    def _resolve_task_type(self, context: dict[str, Any]) -> TaskType:
        task_type = context.get("task_type", self.get_config("task_type", TaskType.CLASSIFICATION))
        return self._coerce_task_type(task_type)

    def _coerce_task_type(self, task_type: TaskType | str) -> TaskType:
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        if not isinstance(task_type, TaskType):
            raise TypeError("task_type must be TaskType or a valid task_type string.")
        return task_type
