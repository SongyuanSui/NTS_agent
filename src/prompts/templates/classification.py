"""Prompt templates for time-series classification."""

SYSTEM_PROMPT = """You are a careful time-series classification assistant.
Use the input series, metadata, and any retrieved labeled examples to infer the most likely class.
Return only valid JSON with the requested keys."""

USER_TEMPLATE = """Task: classify the query time series.

Task description:
{task_description}

Allowed labels:
{label_space}

Query:
{query_block}

Retrieved examples:
{retrieved_block}

Return JSON with this schema:
{{
  "prediction": "<one allowed label>",
  "confidence": <number from 0 to 1>,
  "reasoning": "<brief evidence-based explanation>"
}}"""
