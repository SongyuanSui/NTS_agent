from __future__ import annotations

from typing import Any, Callable, Optional

from experiments.experiment_base import BaseExperiment, ExperimentResult


class AblationExperiment(BaseExperiment):
    """Run named experiment variants supplied by the caller."""

    def run(self, context: Optional[dict[str, Any]] = None) -> ExperimentResult:
        context = {} if context is None else dict(context)
        variants = context.get("variants", {})
        if not isinstance(variants, dict) or not variants:
            raise ValueError("context['variants'] must be a non-empty dict.")

        results: dict[str, Any] = {}
        for name, runner in variants.items():
            if not callable(runner):
                raise TypeError(f"variant {name!r} must be callable.")
            results[str(name)] = runner()

        reducer: Callable[[dict[str, Any]], dict[str, Any]] | None = context.get("reducer")
        metrics = reducer(results) if reducer is not None else {}
        return ExperimentResult(name=self.name, metrics=metrics, metadata={"variants": results})
