from __future__ import annotations
import os

from .client_base import LLMConfig
from .openai_client import OpenAIClient

# Alibaba DashScope exposes an OpenAI-compatible endpoint.
_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenClient(OpenAIClient):
    """
    Client for Qwen (Alibaba) models.

    Hosted via DashScope:
        client = QwenClient(
            config=LLMConfig(model="qwen-max"),
            api_key=os.environ["DASHSCOPE_API_KEY"],
        )

    Self-hosted via SGLang:
        client = QwenClient(
            config=LLMConfig(model="Qwen/Qwen2.5-72B-Instruct"),
            api_key="EMPTY",
            base_url="http://localhost:30000/v1",
        )
    """

    _DEFAULT_MODEL_HOSTED = "qwen-max"
    _DEFAULT_MODEL_LOCAL  = "Qwen/Qwen2.5-72B-Instruct"

    def __init__(
        self,
        config: LLMConfig,
        api_key: str | None = None,
        base_url: str | None = None,
        use_sglang: bool = False,
    ):
        if use_sglang:
            resolved_key = api_key or "EMPTY"
            resolved_url = base_url or "http://localhost:30000/v1"
        else:
            resolved_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
            resolved_url = base_url or _QWEN_BASE_URL

        super().__init__(config=config, api_key=resolved_key, base_url=resolved_url)