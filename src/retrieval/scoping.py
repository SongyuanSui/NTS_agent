"""Retrieval scoping: restrict candidate memory before retrieval.

This is a cross-cutting capability, independent of the representation view
(ts / stat / text / hybrid). A pipeline enables it via the ``retrieval_scope``
config so that any retrieval agent used on that task retrieves only within the
chosen scope. Currently supports restricting candidates to the query's own
source file/run ("same_file"), which realizes the per-file / within-run setting
where cross-file neighbours are excluded.
"""

from __future__ import annotations

import re
from typing import Any

_WINDOW_SUFFIX = re.compile(r"__w\d+_\d+$")


def group_key(sample_id: Any, metadata: Any) -> str:
    """Identify which file/run a window belongs to.

    Prefers metadata['source_file']; falls back to the sample_id with the
    ``__w<start>_<end>`` window suffix stripped (e.g. 'valve1__8').
    """
    if isinstance(metadata, dict):
        src = metadata.get("source_file")
        if src:
            return str(src)
    return _WINDOW_SUFFIX.sub("", str(sample_id))


def scope_memory_bank(memory_bank: Any, query: Any, mode: str | None) -> Any:
    """Return a memory bank restricted according to ``mode``.

    - None / "" / "global": returned unchanged.
    - "same_file": only entries whose group_key matches the query's group_key.

    The result exposes ``get_all()`` (a MemoryBank) so every retriever handles it
    identically, regardless of representation view.
    """
    if memory_bank is None or mode in (None, "", "global"):
        return memory_bank
    if mode != "same_file":
        raise ValueError(f"Unknown retrieval_scope: {mode!r}")

    entries = memory_bank.get_all() if hasattr(memory_bank, "get_all") else list(memory_bank)
    query_key = group_key(
        getattr(query.sample, "sample_id", ""),
        getattr(query.sample, "metadata", None),
    )
    filtered = [
        e
        for e in entries
        if group_key(getattr(e, "sample_id", ""), getattr(e, "metadata", None)) == query_key
    ]

    from memory.memory_bank import MemoryBank

    return MemoryBank(entries=filtered)
