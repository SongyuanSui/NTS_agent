from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pipelines.hooks import HookManager


@dataclass(slots=True)
class AgentGraphNode:
    """One executable node in a simple sequential agent graph."""

    name: str
    component: Any
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("node name must be a non-empty string.")
        if self.output_key is None:
            self.output_key = self.name
        if self.context is None:
            self.context = {}
        if not isinstance(self.context, dict):
            self.context = dict(self.context)


class AgentGraph:
    """
    Minimal ordered graph executor for pipeline stages.

    It intentionally models the current project need: deterministic sequential
    execution of agents/components with named intermediate outputs.
    """

    def __init__(
        self,
        nodes: Optional[list[AgentGraphNode]] = None,
        hooks: Optional[HookManager] = None,
    ) -> None:
        self._nodes: list[AgentGraphNode] = []
        self.hooks = hooks or HookManager()
        for node in nodes or []:
            self.add_node(node)

    @property
    def nodes(self) -> list[AgentGraphNode]:
        return list(self._nodes)

    def add_node(self, node: AgentGraphNode | str, component: Any = None, **kwargs: Any) -> None:
        if isinstance(node, str):
            node = AgentGraphNode(name=node, component=component, **kwargs)
        if not isinstance(node, AgentGraphNode):
            raise TypeError("node must be an AgentGraphNode or node name.")
        self._nodes.append(node)

    def run(
        self,
        initial_input: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        context = {} if context is None else dict(context)
        state: dict[str, Any] = {"input": initial_input}
        current = initial_input

        for node in self._nodes:
            node_input = state[node.input_key] if node.input_key is not None else current
            node_context = {**context, **node.context, "stage": node.name}
            self.hooks.before_stage(node.name, node_input, node_context)
            try:
                if hasattr(node.component, "run"):
                    node_output = node.component.run(node_input, context=node_context)
                elif callable(node.component):
                    node_output = node.component(node_input, context=node_context)
                else:
                    raise TypeError(
                        f"Graph node '{node.name}' component must be callable or expose run(...)."
                    )
            except Exception as exc:
                self.hooks.on_error(node.name, exc, node_context)
                raise

            state[node.output_key or node.name] = node_output
            current = node_output
            self.hooks.after_stage(node.name, node_output, node_context)

        state["output"] = current
        return state
