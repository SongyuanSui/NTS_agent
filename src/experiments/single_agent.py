from __future__ import annotations

from typing import Any, Optional

from experiments.experiment_base import BaseExperiment, ExperimentResult


class SingleAgentExperiment(BaseExperiment):
    """Run one agent over provided input data."""

    def run(self, context: Optional[dict[str, Any]] = None) -> ExperimentResult:
        context = {} if context is None else dict(context)
        agent = context.get("agent")
        input_data = context.get("input_data")
        if agent is None or not hasattr(agent, "run"):
            raise ValueError("context['agent'] with run(...) is required.")
        output = agent.run(input_data, context=context.get("agent_context", {}))
        return ExperimentResult(
            name=self.name,
            metrics={},
            metadata={"output": output},
        )
