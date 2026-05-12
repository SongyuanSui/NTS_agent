from __future__ import annotations
import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any

from .client_base import LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class ParsedResponse:
    raw_text: str
    parsed: Any | None = None       # dict / list / scalar after extraction
    parse_success: bool = False
    parse_error: str | None = None


class ResponseParser:
    """
    Task-agnostic text extraction utilities.

    Sits between the raw LLMResponse and task-specific output_parsers.
    Handles the messy reality that models wrap answers in markdown,
    add preamble, or emit partial JSON.
    """

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def parse_json(self, response: LLMResponse) -> ParsedResponse:
        """
        Extract a JSON object or array from the response.
        Tries (in order):
          1. ```json ... ``` fence
          2. ``` ... ``` fence
          3. First {...} or [...] span in the raw text
          4. The whole text
        """
        text = response.content.strip()

        for candidate in self._json_candidates(text):
            try:
                parsed = json.loads(candidate)
                return ParsedResponse(raw_text=text, parsed=parsed, parse_success=True)
            except json.JSONDecodeError:
                continue

        logger.warning("JSON parse failed for response: %s", text[:200])
        return ParsedResponse(
            raw_text=text,
            parse_success=False,
            parse_error="No valid JSON found in response",
        )

    # ------------------------------------------------------------------
    # Label / classification
    # ------------------------------------------------------------------

    def parse_label(
        self,
        response: LLMResponse,
        label_space: list[str],
        case_sensitive: bool = False,
    ) -> ParsedResponse:
        """
        Find the first occurrence of any label from label_space in the text.
        Falls back to longest-prefix match.
        """
        text = response.content.strip()
        compare = text if case_sensitive else text.lower()
        labels = label_space if case_sensitive else [l.lower() for l in label_space]

        for original, normalized in zip(label_space, labels):
            if normalized in compare:
                return ParsedResponse(raw_text=text, parsed=original, parse_success=True)

        logger.warning(
            "Label parse failed. label_space=%s, response=%s", label_space, text[:200]
        )
        return ParsedResponse(
            raw_text=text,
            parse_success=False,
            parse_error=f"None of {label_space} found in response",
        )

    # ------------------------------------------------------------------
    # Numeric
    # ------------------------------------------------------------------

    def parse_numeric(self, response: LLMResponse) -> ParsedResponse:
        """Extract the first integer or float in the text."""
        text = response.content.strip()
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            value = float(match.group()) if "." in match.group() else int(match.group())
            return ParsedResponse(raw_text=text, parsed=value, parse_success=True)

        return ParsedResponse(
            raw_text=text,
            parse_success=False,
            parse_error="No numeric value found in response",
        )

    # ------------------------------------------------------------------
    # List of values
    # ------------------------------------------------------------------

    def parse_list(self, response: LLMResponse) -> ParsedResponse:
        """
        Extract a list. Tries JSON first, then newline/comma-separated text.
        """
        # Try JSON array first
        json_result = self.parse_json(response)
        if json_result.parse_success and isinstance(json_result.parsed, list):
            return json_result

        # Fall back to splitting on newlines or commas
        text = response.content.strip()
        # Strip bullet/numbering prefixes like "1. ", "- ", "* "
        lines = re.split(r"\n|,", text)
        items = [re.sub(r"^\s*[-*\d.]+\s*", "", l).strip() for l in lines]
        items = [i for i in items if i]

        if items:
            return ParsedResponse(raw_text=text, parsed=items, parse_success=True)

        return ParsedResponse(
            raw_text=text,
            parse_success=False,
            parse_error="Could not extract list from response",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json_candidates(text: str) -> list[str]:
        candidates: list[str] = []

        # ```json ... ``` fence
        m = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            candidates.append(m.group(1).strip())

        # ``` ... ``` fence (no language tag)
        m = re.search(r"```\s*([\s\S]*?)```", text)
        if m:
            candidates.append(m.group(1).strip())

        # First { ... } span
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            candidates.append(m.group(0))

        # First [ ... ] span
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            candidates.append(m.group(0))

        # Whole text as last resort
        candidates.append(text)
        return candidates