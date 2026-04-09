# src/pipelines/pipeline_base.py

from __future__ import annotations

from abc import ABC
from typing import Any, Iterable, Optional

from core.interfaces import BasePipelineInterface
from core.schemas import (
    BatchPredictionRecord,
    PipelineResult,
    PredictionRecord,
    TimeSeriesSample,
)


class BasePipeline(BasePipelineInterface, ABC):
    """
    Reusable base class for all pipelines.

    Responsibilities
    ----------------
    A concrete pipeline is responsible for orchestrating components such as:
    - task
    - channel_decomposer
    - representation agents
    - retrieval agents
    - reasoner
    - aggregator

    This base class provides:
    - component registration and access
    - validation helpers
    - single-sample / batch execution skeleton
    - common description and config utilities

    Design goals
    ------------
    - keep orchestration logic centralized
    - make end-to-end execution and debugging consistent
    - allow future extension to memory-building and ablation pipelines
    """

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        components: Optional[dict[str, Any]] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, config=config)
        self._components = components or {}
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def components(self) -> dict[str, Any]:
        return self._components

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def add_component(self, key: str, component: Any) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{self.name}: component key must be a non-empty string.")
        self._components[key] = component

    def get_component(self, key: str, default: Any = None) -> Any:
        return self._components.get(key, default)

    def require_component(self, key: str) -> Any:
        if key not in self._components:
            raise KeyError(f"{self.name}: required component '{key}' is missing.")
        return self._components[key]

    def has_component(self, key: str) -> bool:
        return key in self._components

    def normalize_context(self, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if context is None:
            return {}
        if not isinstance(context, dict):
            raise TypeError(f"{self.name}: context must be a dict or None.")
        return context

    def validate_sample(self, sample: TimeSeriesSample) -> None:
        if not isinstance(sample, TimeSeriesSample):
            raise TypeError(
                f"{self.name}: sample must be a TimeSeriesSample, "
                f"but got {type(sample).__name__}."
            )

    def validate_samples(self, samples: Iterable[TimeSeriesSample]) -> list[TimeSeriesSample]:
        if samples is None:
            raise ValueError(f"{self.name}: samples must not be None.")

        sample_list = list(samples)
        for sample in sample_list:
            self.validate_sample(sample)
        return sample_list

    def validate_components(self) -> None:
        """
        Default pipeline-level component validation.

        Concrete pipelines should usually override this and check their exact
        required component set. This base implementation only checks that the
        component container exists.
        """
        if not isinstance(self._components, dict):
            raise TypeError(f"{self.name}: components must be stored as a dict.")

    def describe(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"enabled={self.enabled}, "
            f"components={sorted(self.components.keys())}, "
            f"config_keys={sorted(self.config.keys())}"
            f")"
        )

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def require_config(self, key: str) -> Any:
        if key not in self.config:
            raise KeyError(f"{self.name}: required config key '{key}' is missing.")
        return self.config[key]

    def pre_run(
        self,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Shared pre-run hook for single-sample execution.
        """
        if not self.enabled:
            raise RuntimeError(f"{self.name}: pipeline is disabled.")

        self.validate_components()
        self.validate_sample(sample)

    def post_run(
        self,
        result: PipelineResult,
        context: Optional[dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Shared post-run hook for single-sample execution.
        """
        return result

    def run(
        self,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Standard single-sample execution entrypoint.

        Execution flow:
        1. normalize context
        2. pre_run(...)
        3. _run_impl(...)
        4. post_run(...)
        """
        context = self.normalize_context(context)
        self.pre_run(sample=sample, context=context)
        result = self._run_impl(sample=sample, context=context)
        return self.post_run(result=result, context=context)

    def run_batch(
        self,
        samples: Iterable[TimeSeriesSample],
        context: Optional[dict[str, Any]] = None,
    ) -> BatchPredictionRecord:
        """
        Default batch execution implementation.

        This intentionally uses repeated single-sample execution first.
        Later you can override it for more efficient batched behavior.
        """
        context = self.normalize_context(context)
        sample_list = self.validate_samples(samples)

        records: list[PredictionRecord] = []
        batch_metadata: dict[str, Any] = {
            "pipeline_name": self.name,
            "num_samples": len(sample_list),
        }

        for sample in sample_list:
            result = self.run(sample=sample, context=context)
            records.append(result.prediction)

        return BatchPredictionRecord(records=records, metadata=batch_metadata)

    def _run_impl(
        self,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Concrete end-to-end execution logic implemented by subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _run_impl(...)."
        )