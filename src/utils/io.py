# src/utils/io.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist and return it as a Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Ensure the parent directory of a file path exists.
    Return the original file path as a Path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    """
    Read a text file.
    """
    path = Path(path)
    return path.read_text(encoding=encoding)


def write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """
    Write a text file, creating parent directories if needed.
    """
    path = ensure_parent_dir(path)
    path.write_text(text, encoding=encoding)
    return path


def read_json(path: str | Path, encoding: str = "utf-8") -> Any:
    """
    Read a JSON file and return the parsed object.
    """
    path = Path(path)
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def write_json(
    path: str | Path,
    data: Any,
    indent: int = 2,
    ensure_ascii: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """
    Write a JSON file, creating parent directories if needed.
    """
    path = ensure_parent_dir(path)
    with open(path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    return path


def read_jsonl(path: str | Path, encoding: str = "utf-8") -> list[Any]:
    """
    Read a JSONL file into a list of Python objects.
    """
    path = Path(path)
    records: list[Any] = []
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(
    path: str | Path,
    records: list[Any],
    ensure_ascii: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """
    Write a list of objects to a JSONL file.
    """
    path = ensure_parent_dir(path)
    with open(path, "w", encoding=encoding) as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")
    return path