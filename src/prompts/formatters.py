# src/prompts/formatters.py

from __future__ import annotations

import math
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Any, Iterable, Sequence

import numpy as np

from core.schemas import ChannelData
from retrieval.schemas import RetrievedExample, RetrievedSet


def format_float_for_llm(
    value: float | int,
    decimals: int = 3,
    mode: str = "round",
    strip_trailing_zeros: bool = True,
) -> str:
    """
    Format one numeric value for LLM-facing text.

    Parameters
    ----------
    value:
        Numeric value to format.
    decimals:
        Number of digits after the decimal point.
    mode:
        Either:
        - "round": round to nearest using HALF_UP
        - "truncate": cut off extra decimals without rounding
    strip_trailing_zeros:
        Whether to remove unnecessary trailing zeros and a dangling decimal point.

    Examples
    --------
    >>> format_float_for_llm(1.23456, decimals=3, mode="round")
    '1.235'
    >>> format_float_for_llm(1.23456, decimals=3, mode="truncate")
    '1.234'
    >>> format_float_for_llm(2.0, decimals=3)
    '2'
    """
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"value must be numeric, but got {type(value).__name__}.")

    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer.")

    if mode not in {"round", "truncate"}:
        raise ValueError("mode must be either 'round' or 'truncate'.")

    value = float(value)

    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"

    quant = Decimal("1") if decimals == 0 else Decimal(f"1e-{decimals}")
    decimal_value = Decimal(str(value))

    if mode == "round":
        formatted = decimal_value.quantize(quant, rounding=ROUND_HALF_UP)
    else:
        formatted = decimal_value.quantize(quant, rounding=ROUND_DOWN)

    text = format(formatted, "f")

    if strip_trailing_zeros and "." in text:
        text = text.rstrip("0").rstrip(".")
        if text == "-0":
            text = "0"

    return text


def format_series_for_llm(
    values: Sequence[float] | np.ndarray,
    decimals: int = 3,
    mode: str = "round",
    separator: str = ", ",
    left_bracket: str = "[",
    right_bracket: str = "]",
    max_items: int | None = None,
    include_length_suffix: bool = False,
) -> str:
    """
    Format a 1D series into a compact LLM-readable string.

    Parameters
    ----------
    values:
        1D numeric sequence.
    decimals:
        Number of digits after the decimal point.
    mode:
        "round" or "truncate".
    separator:
        Separator between elements.
    max_items:
        If provided, only show the first `max_items` elements.
    include_length_suffix:
        If True and truncation happens, append a short suffix like:
        "... (total_length=128)"

    Examples
    --------
    >>> format_series_for_llm([1.23456, 2.34567], decimals=2)
    '[1.23, 2.35]'
    """
    arr = np.asarray(values, dtype=float)

    if arr.ndim != 1:
        raise ValueError(f"values must be 1D, but got ndim={arr.ndim}.")

    if max_items is not None:
        if not isinstance(max_items, int) or max_items <= 0:
            raise ValueError("max_items must be a positive integer or None.")

    original_length = len(arr)

    if max_items is not None and original_length > max_items:
        arr = arr[:max_items]
        truncated = True
    else:
        truncated = False

    formatted_values = [
        format_float_for_llm(v, decimals=decimals, mode=mode) for v in arr.tolist()
    ]
    text = f"{left_bracket}{separator.join(formatted_values)}{right_bracket}"

    if truncated:
        if include_length_suffix:
            text += f" ... (total_length={original_length})"
        else:
            text += " ..."

    return text


def format_channel_for_llm(
    channel: ChannelData,
    decimals: int = 3,
    mode: str = "round",
    max_items: int | None = None,
    include_score: bool = False,
) -> str:
    """
    Format one ChannelData object into a readable LLM-facing string.

    Example
    -------
    Channel 0: [0.12, 0.35, 0.91, ...]
    """
    if not isinstance(channel, ChannelData):
        raise TypeError("channel must be a ChannelData object.")

    series_text = format_series_for_llm(
        channel.values,
        decimals=decimals,
        mode=mode,
        max_items=max_items,
        include_length_suffix=True,
    )

    parts = [f"Channel {channel.channel_id}: {series_text}"]

    if include_score and channel.score is not None:
        score_text = format_float_for_llm(channel.score, decimals=4, mode="round")
        parts.append(f"(score={score_text})")

    return " ".join(parts)


def format_channels_for_llm(
    channels: Sequence[ChannelData],
    decimals: int = 3,
    mode: str = "round",
    max_items_per_channel: int | None = None,
    include_score: bool = False,
) -> str:
    """
    Format multiple channels into a multi-line string.

    Example
    -------
    Channel 0: [...]
    Channel 2: [...]
    Channel 5: [...]
    """
    if not isinstance(channels, Sequence):
        raise TypeError("channels must be a sequence of ChannelData.")

    lines = [
        format_channel_for_llm(
            channel=channel,
            decimals=decimals,
            mode=mode,
            max_items=max_items_per_channel,
            include_score=include_score,
        )
        for channel in channels
    ]
    return "\n".join(lines)


def format_retrieved_example_for_llm(
    example: RetrievedExample,
    decimals: int = 3,
    mode: str = "round",
    max_items: int | None = None,
    include_payload: bool = True,
) -> str:
    """
    Format one RetrievedExample into a readable text block.

    payload rendering policy
    ------------------------
    - ndarray / list-like 1D numeric payload -> render as series
    - str payload -> use directly
    - dict payload -> key=value pairs
    - otherwise -> str(payload)
    """
    if not isinstance(example, RetrievedExample):
        raise TypeError("example must be a RetrievedExample.")

    score_text = format_float_for_llm(
        example.score.value,
        decimals=4,
        mode="round",
    )

    header = (
        f"Sample {example.sample_id} | "
        f"Label: {example.label} | "
        f"Channel: {example.channel_id} | "
        f"View: {example.representation_type.value} | "
        f"Score: {score_text}"
    )

    if not include_payload:
        return header

    payload_text = _format_payload_for_llm(
        payload=example.payload,
        decimals=decimals,
        mode=mode,
        max_items=max_items,
    )

    return f"{header}\nPayload: {payload_text}"


def format_retrieved_set_for_llm(
    retrieved_set: RetrievedSet,
    decimals: int = 3,
    mode: str = "round",
    max_items_per_payload: int | None = None,
    include_payload: bool = True,
) -> str:
    """
    Format a RetrievedSet into a multi-example text block suitable for prompts.
    """
    if not isinstance(retrieved_set, RetrievedSet):
        raise TypeError("retrieved_set must be a RetrievedSet.")

    if retrieved_set.is_empty:
        return f"(No retrieved examples for query {retrieved_set.query_id})"

    blocks = []
    for idx, example in enumerate(retrieved_set.examples, start=1):
        block = format_retrieved_example_for_llm(
            example=example,
            decimals=decimals,
            mode=mode,
            max_items=max_items_per_payload,
            include_payload=include_payload,
        )
        blocks.append(f"Example {idx}:\n{block}")

    return "\n\n".join(blocks)


def _format_payload_for_llm(
    payload: Any,
    decimals: int = 3,
    mode: str = "round",
    max_items: int | None = None,
) -> str:
    """
    Internal helper for payload rendering.

    Supported payload categories
    ----------------------------
    - 1D numeric ndarray
    - numeric list/tuple
    - string
    - dict
    - fallback: str(payload)
    """
    if payload is None:
        return "None"

    if isinstance(payload, str):
        return payload

    if isinstance(payload, np.ndarray):
        if payload.ndim == 1:
            return format_series_for_llm(
                payload,
                decimals=decimals,
                mode=mode,
                max_items=max_items,
                include_length_suffix=True,
            )
        return str(payload.tolist())

    if isinstance(payload, (list, tuple)):
        try:
            arr = np.asarray(payload, dtype=float)
            if arr.ndim == 1:
                return format_series_for_llm(
                    arr,
                    decimals=decimals,
                    mode=mode,
                    max_items=max_items,
                    include_length_suffix=True,
                )
        except (TypeError, ValueError):
            pass
        return str(payload)

    if isinstance(payload, dict):
        items = []
        for key, value in payload.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                value_text = format_float_for_llm(
                    value,
                    decimals=decimals,
                    mode=mode,
                )
            else:
                value_text = str(value)
            items.append(f"{key}={value_text}")
        return "{ " + ", ".join(items) + " }"

    return str(payload)