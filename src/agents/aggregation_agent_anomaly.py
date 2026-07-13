"""Registration shim for the anomaly aggregation component.

The inference runner maps a component name (e.g. ``aggregation_agent_anomaly``)
to a module ``agents.<name>`` and imports it so the ``@AGENT_REGISTRY.decorator``
side effect runs. The anomaly pipeline references a dedicated config file
(``configs/agents/aggregation_agent_anomaly.yaml``) to enable ``any_positive``
aggregation without touching the shared classification config, but it reuses the
existing :class:`AggregationAgent` class. Importing it here guarantees the class
is registered under its ``name`` ("aggregation_agent") when this module is loaded.
"""

from __future__ import annotations

from agents.aggregation_agent import AggregationAgent  # noqa: F401
