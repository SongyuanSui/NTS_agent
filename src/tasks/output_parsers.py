from __future__ import annotations

from typing import Any


PREDICTION_KEYS = ("prediction", "decision", "label", "answer", "output")
CONFIDENCE_KEYS = ("confidence", "score", "probability")


def parse_prediction_value(raw_output: Any) -> Any:
    """Extract the prediction-like value from common downstream output shapes."""
    if isinstance(raw_output, dict):
        for key in PREDICTION_KEYS:
            if key in raw_output:
                return raw_output[key]
    if hasattr(raw_output, "prediction"):
        return getattr(raw_output, "prediction")
    return raw_output


def parse_confidence(raw_output: Any) -> float | None:
    """Extract an optional confidence score from common output shapes."""
    value = None
    if isinstance(raw_output, dict):
        for key in CONFIDENCE_KEYS:
            if key in raw_output:
                value = raw_output[key]
                break
    elif hasattr(raw_output, "confidence"):
        value = getattr(raw_output, "confidence")

    if value is None:
        return None
    confidence = float(value)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be within [0.0, 1.0].")
    return confidence


def parse_binary_anomaly(raw_output: Any) -> str:
    """Normalize common anomaly outputs into 'normal' or 'anomaly'."""
    value = parse_prediction_value(raw_output)
    if isinstance(value, bool):
        return "anomaly" if value else "normal"
    if isinstance(value, (int, float)):
        return "anomaly" if int(value) == 1 else "normal"

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "anomalous", "anomaly"}:
        return "anomaly"
    if text in {"0", "false", "no", "normal", "nominal"}:
        return "normal"
    raise ValueError(f"Cannot parse anomaly output: {value!r}")
