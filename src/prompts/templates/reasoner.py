"""Prompt templates for time-series reasoning (decision-making on retrieved neighbors)."""

SYSTEM_PROMPT = """You are a time-series reasoning assistant.

Your job:
- Read the query representations, retrieved neighbors, and task context.
- Choose the best task-specific output.

Output rules:
- Output exactly one JSON object and nothing else.
- Use the schema from the user template.
- If you cannot answer, output exactly {}.
- Do not output chain-of-thought, commentary, markdown, or code fences.
- Keep any evidence only inside the `reasoning` field.

Strictly follow the output rules. Any extra text is invalid."""

USER_TEMPLATE = """Channel ID: {channel_id}

Query:
  Raw Series (truncated): {query_raw}
  Text Summary: {query_text}
  TSFEL Feature Statistics: {query_stats}

Retrieved Neighbors (ranked by retrieval score):
{neighbors_block}

Task Type: {task_type}
Task Target: {task_target}
Task: {task_name}
Allowed Labels: {label_space}

INSTRUCTIONS:
- Produce one best task output.
- Choose one `label` from the Allowed Labels when the task is label-based.
- Provide `confidence` as a number between 0 and 1.
- Keep `reasoning` short and factual.
- Do not explain your chain of thought.


OUTPUT EXAMPLE (must match exactly this JSON shape; do NOT output any other text):
{{"label": "<one allowed label>", "confidence": 0.75, "reasoning": "brief evidence"}}

If you cannot answer, output exactly: {{}}

Remember: ONLY output the single JSON object and nothing else."""

NEIGHBOR_TEMPLATE = """- ID: {sample_id}
  Label: {label}
  Score: {score}
  Raw Series (truncated): {raw_series}
  Text Summary: {text_summary}
  TSFEL Feature Statistics: {stats}"""
