"""Prompt templates for time-series forecasting/prediction."""

SYSTEM_PROMPT = """You are a careful time-series prediction assistant.
Use the observed input series, metadata, and retrieved examples to forecast the requested target.
Return only valid JSON with the requested keys."""

USER_TEMPLATE = """Task: predict the future behavior or target for the query time series.

Task description:
{task_description}

Prediction metadata:
{prediction_metadata}

Query:
{query_block}

Retrieved examples:
{retrieved_block}

Return JSON with this schema:
{{
  "prediction": <predicted value, label, or sequence>,
  "confidence": <number from 0 to 1>,
  "reasoning": "<brief evidence-based explanation>"
}}"""
