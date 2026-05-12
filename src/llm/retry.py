from __future__ import annotations
import time
import random
import logging
import functools
from dataclasses import dataclass, field
from typing import Callable, TypeVar, Any

import openai

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0        # seconds before first retry
    max_delay: float = 30.0        # cap on backoff
    exponential_base: float = 2.0
    jitter: bool = True            # add ±20 % randomness to avoid thundering herd
    # Exceptions that warrant a retry (provider-agnostic names resolved below)
    retryable_status_codes: set[int] = field(default_factory=lambda: {429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception, cfg: RetryConfig) -> bool:
    """Return True if the exception is transient and worth retrying."""
    # openai SDK specific
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APITimeoutError):
        return True
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in cfg.retryable_status_codes
    # Generic network errors
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return False


def _is_fatal(exc: Exception) -> bool:
    """Return True if retrying would never help."""
    if isinstance(exc, openai.AuthenticationError):
        return True
    if isinstance(exc, openai.BadRequestError):
        return True
    if isinstance(exc, openai.NotFoundError):
        return True
    return False


# ---------------------------------------------------------------------------
# Backoff calculation
# ---------------------------------------------------------------------------

def _backoff_seconds(attempt: int, cfg: RetryConfig) -> float:
    """Exponential backoff with optional jitter. `attempt` is 0-indexed."""
    delay = min(cfg.base_delay * (cfg.exponential_base ** attempt), cfg.max_delay)
    if cfg.jitter:
        delay *= 0.8 + random.random() * 0.4   # ±20 %
    return delay


# ---------------------------------------------------------------------------
# Public decorator / wrapper
# ---------------------------------------------------------------------------

def with_retry(retry_cfg: RetryConfig | None = None) -> Callable[[F], F]:
    """
    Decorator that wraps an LLMClient.complete() call with retry logic.

    Usage:
        @with_retry(RetryConfig(max_attempts=5))
        def complete(self, request): ...

    Or wrap an arbitrary callable:
        safe_complete = with_retry(cfg)(client.complete)
        response = safe_complete(request)
    """
    cfg = retry_cfg or RetryConfig()

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None

            for attempt in range(cfg.max_attempts):
                try:
                    return fn(*args, **kwargs)

                except Exception as exc:
                    last_exc = exc

                    if _is_fatal(exc):
                        logger.error("Fatal LLM error (no retry): %s", exc)
                        raise

                    if not _is_retryable(exc, cfg):
                        logger.error("Non-retryable LLM error: %s", exc)
                        raise

                    if attempt + 1 == cfg.max_attempts:
                        break

                    delay = _backoff_seconds(attempt, cfg)
                    logger.warning(
                        "Retryable LLM error (attempt %d/%d, retry in %.1fs): %s",
                        attempt + 1, cfg.max_attempts, delay, exc,
                    )
                    time.sleep(delay)

            raise RuntimeError(
                f"LLM call failed after {cfg.max_attempts} attempts"
            ) from last_exc

        return wrapper  # type: ignore[return-value]
    return decorator