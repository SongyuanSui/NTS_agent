from __future__ import annotations
import time
import logging
from typing import Any

import openai
from openai import OpenAI, APITimeoutError, RateLimitError, APIStatusError

from .client_base import LLMClient, LLMConfig, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """
    Client for OpenAI and any OpenAI-compatible endpoint
    (vLLM, SGLang, Together AI, Fireworks, etc.).

    Example — SGLang local server:
        client = OpenAIClient(
            config=LLMConfig(model="meta-llama/Llama-3-8b-instruct"),
            api_key="EMPTY",
            base_url="http://localhost:30000/v1",
        )
    """

    def __init__(
        self,
        config: LLMConfig,
        api_key: str,
        base_url: str | None = None,   # None → official OpenAI endpoint
    ):
        super().__init__(config)
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=config.timeout,
        )
        self._base_url = base_url

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        cfg = self._resolve_config(request)
        messages = self._build_messages(request)

        logger.debug(
            "LLM request",
            extra={"model": cfg.model, "n_messages": len(messages), **request.metadata},
        )

        t0 = time.perf_counter()
        raw = self._client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            **cfg.extra_kwargs,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        response = LLMResponse(
            content=raw.choices[0].message.content or "",
            model=raw.model,
            usage=self._extract_usage(raw),
            raw=raw,
            latency_ms=latency_ms,
        )

        logger.debug(
            "LLM response",
            extra={
                "model": response.model,
                "latency_ms": round(latency_ms, 1),
                "usage": response.usage,
                **request.metadata,
            },
        )
        return response

    def is_available(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_messages(self, request: LLMRequest) -> list[dict]:
        """Prepend system message if provided; return full message list."""
        messages: list[dict] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(request.messages)
        return messages

    @staticmethod
    def _extract_usage(raw: Any) -> dict:
        if raw.usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        u = raw.usage
        return {
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        }