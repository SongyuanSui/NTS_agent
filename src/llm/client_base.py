from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """Provider-agnostic generation parameters."""
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
    timeout: int = 60
    # Pass provider-specific kwargs (e.g. stop sequences, seed) without
    # polluting the typed interface.
    extra_kwargs: dict = field(default_factory=dict)


@dataclass
class LLMRequest:
    """Everything needed for a single completion call."""
    messages: list[dict]          # [{"role": "user"|"assistant"|"system", "content": str}]
    system: str | None = None     # Convenience: prepended as system message if set
    config: LLMConfig | None = None  # Falls back to client default if None
    metadata: dict = field(default_factory=dict)  # Passed through to logs; never sent to API


@dataclass
class LLMResponse:
    """Normalised output from any provider."""
    content: str
    model: str
    usage: dict                   # {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
    raw: Any                      # Original SDK response; keep for debugging
    latency_ms: float


class LLMClient(ABC):
    """Abstract base all provider clients must implement."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a request and return a normalised response."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Health-check; used by the factory before routing."""
        ...

    def _resolve_config(self, request: LLMRequest) -> LLMConfig:
        """Request-level config overrides client default."""
        return request.config if request.config is not None else self.config