"""Prompt templates for anomaly-window detection."""

SYSTEM_PROMPT = """You are a careful time-series anomaly detection assistant.
Decide whether the query window is normal or anomalous using the series, metadata, and retrieved examples.
Return only valid JSON with the requested keys."""

USER_TEMPLATE = """Task: determine whether the query time-series window is anomalous.

Task description:
{task_description}

Anomaly metadata:
{anomaly_metadata}

Allowed labels:
{label_space}

Query:
{query_block}

Retrieved examples:
{retrieved_block}

Return JSON with this schema:
{{
  "prediction": "<normal or anomaly label>",
  "confidence": <number from 0 to 1>,
  "reasoning": "<brief evidence-based explanation>"
}}"""
