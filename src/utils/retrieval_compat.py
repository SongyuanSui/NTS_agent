from __future__ import annotations

from typing import Any


def unwrap_payload(p: Any) -> Any:
    """Compatibility helper: unwrap a provenance-wrapped payload.

    If `p` is of the form {"channel_id": <int>, "view": <view>}, return the
    inner `view`. Otherwise return `p` unchanged.
    """
    if isinstance(p, dict) and "channel_id" in p and "view" in p:
        return p["view"]
    return p


def unwrap_record_payload(record) -> Any:
    """Convenience wrapper for record-like objects with a `.payload` attribute."""
    return unwrap_payload(getattr(record, "payload", None))
