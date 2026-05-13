"""Prompt templates for natural-language time-series summaries."""

SYSTEM_PROMPT = """You are a concise time-series summarization assistant.
Describe the series shape, scale, trend, variability, and notable events.
Return only valid JSON with the requested keys."""

USER_TEMPLATE = """Task: summarize the query time series for retrieval and downstream reasoning.

Summary style:
{summary_style}

Query:
{query_block}

Return JSON with this schema:
{{
  "summary": "<concise time-series summary>",
  "key_patterns": ["<pattern>", "..."],
  "reasoning": "<brief explanation of the summary>"
}}"""
