# src/pipelines/pipeline_base.py

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, Iterable, Optional

from core.interfaces import BasePipelineInterface
from core.schemas import (
    BatchPredictionRecord,
    PipelineResult,
    PredictionRecord,
    TimeSeriesSample,
)
from ts_logging.event_log import EventLogger


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
    - optional logger / event logger access through runtime context

    Design goals
    ------------
    - keep orchestration logic centralized
    - make end-to-end execution and debugging consistent
    - allow future extension to memory-building and ablation pipelines
    - avoid owning path / logger creation logic inside pipeline classes
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

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------
    def get_logger(self, context: Optional[dict[str, Any]] = None) -> Optional[logging.Logger]:
        """
        Return a human-readable logger from context if available.
        """
        if context is None:
            return None
        logger = context.get("logger")
        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError(f"{self.name}: context['logger'] must be a logging.Logger.")
        return logger

    def get_event_logger(self, context: Optional[dict[str, Any]] = None) -> Optional[EventLogger]:
        """
        Return a structured event logger from context if available.
        """
        if context is None:
            return None
        event_logger = context.get("event_logger")
        if event_logger is not None and not isinstance(event_logger, EventLogger):
            raise TypeError(
                f"{self.name}: context['event_logger'] must be an EventLogger."
            )
        return event_logger

    def log_info(self, context: Optional[dict[str, Any]], message: str, *args: Any) -> None:
        logger = self.get_logger(context)
        if logger is not None:
            logger.info(message, *args)

    def log_warning(self, context: Optional[dict[str, Any]], message: str, *args: Any) -> None:
        logger = self.get_logger(context)
        if logger is not None:
            logger.warning(message, *args)

    def log_event(
        self,
        context: Optional[dict[str, Any]],
        event_type: str,
        payload: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        event_logger = self.get_event_logger(context)
        if event_logger is not None:
            event_logger.log_event(event_type=event_type, payload=payload, **kwargs)

    # ------------------------------------------------------------------
    # Execution hooks
    # ------------------------------------------------------------------
    def pre_run(
        self,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            raise RuntimeError(f"{self.name}: pipeline is disabled.")

        self.validate_components()
        self.validate_sample(sample)

        self.log_info(
            context,
            "Starting pipeline '%s' for sample_id=%s",
            self.name,
            sample.sample_id,
        )
        self.log_event(
            context,
            event_type="pipeline_start",
            payload={
                "pipeline_name": self.name,
                "sample_id": sample.sample_id,
            },
        )

    def post_run(
        self,
        result: PipelineResult,
        context: Optional[dict[str, Any]] = None,
    ) -> PipelineResult:
        self.log_info(
            context,
            "Finished pipeline '%s' for sample_id=%s",
            self.name,
            result.prediction.sample_id,
        )
        self.log_event(
            context,
            event_type="pipeline_end",
            payload={
                "pipeline_name": self.name,
                "sample_id": result.prediction.sample_id,
                "prediction": result.prediction.prediction,
                "confidence": result.prediction.confidence,
            },
        )
        return result

    def run(
        self,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Standard single-sample execution entrypoint.

        Execution flow
        --------------
        1. normalize context
        2. pre_run(...)
        3. _run_impl(...)
        4. post_run(...)

        On exception
        ------------
        - emit warning log
        - emit pipeline_error event
        - re-raise the original exception
        """
        context = self.normalize_context(context)

        try:
            self.pre_run(sample=sample, context=context)
            result = self._run_impl(sample=sample, context=context)
            result = self.post_run(result=result, context=context)
        except Exception as exc:
            sample_id = sample.sample_id if isinstance(sample, TimeSeriesSample) else None
            self.log_warning(
                context,
                "Pipeline '%s' failed for sample_id=%s with %s: %s",
                self.name,
                sample_id,
                type(exc).__name__,
                str(exc),
            )
            self.log_event(
                context,
                event_type="pipeline_error",
                payload={
                    "pipeline_name": self.name,
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise

        return result

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

        self.log_info(
            context,
            "Starting batch pipeline '%s' with num_samples=%d",
            self.name,
            len(sample_list),
        )
        self.log_event(
            context,
            event_type="pipeline_batch_start",
            payload={
                "pipeline_name": self.name,
                "num_samples": len(sample_list),
            },
        )

        records: list[PredictionRecord] = []
        batch_metadata: dict[str, Any] = {
            "pipeline_name": self.name,
            "num_samples": len(sample_list),
        }

        try:
            for sample in sample_list:
                result = self.run(sample=sample, context=context)
                records.append(result.prediction)
        except Exception as exc:
            self.log_warning(
                context,
                "Batch pipeline '%s' failed with %s: %s",
                self.name,
                type(exc).__name__,
                str(exc),
            )
            self.log_event(
                context,
                event_type="pipeline_batch_error",
                payload={
                    "pipeline_name": self.name,
                    "num_samples": len(sample_list),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise

        self.log_info(
            context,
            "Finished batch pipeline '%s' with num_samples=%d",
            self.name,
            len(sample_list),
        )
        self.log_event(
            context,
            event_type="pipeline_batch_end",
            payload={
                "pipeline_name": self.name,
                "num_samples": len(sample_list),
            },
        )

        return BatchPredictionRecord(records=records, metadata=batch_metadata)

    def _run_impl(
        self,
        sample: TimeSeriesSample,
        context: Optional[dict[str, Any]] = None,
    ) -> PipelineResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _run_impl(...)."
        )