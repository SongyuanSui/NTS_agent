from __future__ import annotations

from typing import Iterable


def select_feature_groups(
    available_groups: Iterable[str],
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    """
    Select statistical feature groups by include/exclude filters.

    This is intentionally small: feature computation owns the actual TSFEL
    configs, while this helper normalizes user/config choices.
    """
    available = [str(group) for group in available_groups]
    selected = available

    if include is not None:
        include_set = {str(group) for group in include}
        selected = [group for group in selected if group in include_set]

    if exclude is not None:
        exclude_set = {str(group) for group in exclude}
        selected = [group for group in selected if group not in exclude_set]

    return selected
