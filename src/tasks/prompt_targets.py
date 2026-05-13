from __future__ import annotations

from core.enums import TaskType


def get_prompt_target(task_type: TaskType | str, label_space: list[str] | None = None) -> str:
    """Return the default prompt target text for a task type."""
    if isinstance(task_type, str):
        task_type = TaskType(task_type)

    if task_type == TaskType.CLASSIFICATION:
        if label_space:
            return "predict the class label from: [" + ", ".join(label_space) + "]"
        return "predict the class label"
    if task_type == TaskType.PREDICTION:
        return "predict the future target for the input time series"
    if task_type == TaskType.ANOMALY_SEQUENCE:
        return "determine whether the sequence is normal or anomalous"
    if task_type == TaskType.ANOMALY_WINDOW:
        return "determine whether the window is normal or anomalous"
    raise ValueError(f"Unsupported task_type: {task_type!r}")
