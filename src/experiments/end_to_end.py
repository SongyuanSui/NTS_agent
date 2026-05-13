from __future__ import annotations

from typing import Any, Optional

from experiments.experiment_base import BaseExperiment, ExperimentResult


class EndToEndExperiment(BaseExperiment):
    """Run a pipeline over a sample collection and return batch metadata."""

    def run(self, context: Optional[dict[str, Any]] = None) -> ExperimentResult:
        context = {} if context is None else dict(context)
        pipeline = context.get("pipeline")
        samples = context.get("samples")
        if pipeline is None or not hasattr(pipeline, "run_batch"):
            raise ValueError("context['pipeline'] with run_batch(...) is required.")
        if samples is None:
            raise ValueError("context['samples'] is required.")
        batch = pipeline.run_batch(samples, context=context.get("pipeline_context", {}))
        return ExperimentResult(
            name=self.name,
            metrics=dict(getattr(batch, "metadata", {})),
            metadata={"predictions": list(batch.records)},
        )
