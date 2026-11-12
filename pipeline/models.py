"""Shared Pydantic records used by more than one pipeline module.

``DocumentChunk`` and ``Indicator`` are the records of the legacy extraction
step, which the paper-style pipeline keeps as the *novel-indicator pass*
(docs/plan.md Section 4.3). Moved unchanged from ``extraction_utils.py``.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, field_validator


class DocumentChunk(BaseModel):
    doc_id: str
    source_file: str
    chunk_type: str  # "table" | "text"
    page_or_section: Optional[str]
    content: str
    char_count: int


INDICATOR_LEVELS = ("impact", "outcome", "output", "activity", "unclear")
CONFIDENCE_LEVELS = ("high", "medium", "low")


def normalize_indicator_level(value) -> str:
    """Map free-text level values from the LLM onto the plan's enum; anything unrecognised becomes 'unclear'."""
    if value is None:
        return "unclear"
    text = str(value).strip().lower()
    if text in INDICATOR_LEVELS:
        return text
    # e.g. "Intermediate outcome", "Output 2", "Impact indicator", "Result 2 - Indicator 1"
    for lvl in ("impact", "outcome", "output", "activity"):
        if re.search(rf"\b{lvl}s?\b", text):
            return lvl
    if re.search(r"\b(goal|objective)\b", text):
        return "impact"
    return "unclear"


def normalize_confidence(value) -> str:
    if value is None:
        return "low"
    text = str(value).strip().lower()
    return text if text in CONFIDENCE_LEVELS else "low"


class Indicator(BaseModel):
    """One indicator extracted from a document chunk by the legacy extraction prompt."""

    indicator_text: str
    level: str = "unclear"  # impact | outcome | output | activity | unclear
    baseline: Optional[str] = None
    target: Optional[str] = None
    means_of_verification: Optional[str] = None
    sector: Optional[str] = None
    disaggregation: Optional[str] = None
    unit_or_type: Optional[str] = None
    source_confidence: str = "low"  # high | medium | low
    doc_id: Optional[str] = None
    chunk_ref: Optional[str] = None

    @field_validator("level", mode="before")
    @classmethod
    def _coerce_level(cls, v):
        return normalize_indicator_level(v)

    @field_validator("source_confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v):
        return normalize_confidence(v)

    @field_validator("baseline", "target", "means_of_verification", "sector", "disaggregation", "unit_or_type", mode="before")
    @classmethod
    def _coerce_optional_text(cls, v):
        # The LLM often returns numbers (e.g. target: 60) or lists; keep them as strings rather than dropping the row.
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            v = "; ".join(str(x) for x in v if x is not None)
        s = str(v).strip()
        return s if s and s.lower() not in ("null", "none", "n/a", "na") else None

    @field_validator("indicator_text", mode="before")
    @classmethod
    def _coerce_indicator_text(cls, v):
        if v is None:
            raise ValueError("indicator_text is required")
        s = re.sub(r"\s+", " ", str(v)).strip()
        if not s:
            raise ValueError("indicator_text is empty")
        return s
