"""Answer schema of the LLM agent (paper Section 3.4.2 answer format; docs/plan.md Section 4.3).

The paper's answer is <Disclosure, KPI, Topic, Value, Unit, Target, Action>; ``DisclosureAnswer`` mirrors it
and adds ``match_type`` (verbatim / close) for IndiKit.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DisclosureAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indikit_id: str
    disclosure: Literal["present", "absent"]
    match_type: Optional[Literal["verbatim", "close"]] = None
    document_indicator_text: Optional[str] = None
    level: Optional[Literal["impact", "outcome", "output", "unclear"]] = None
    baseline: Optional[str] = None
    target: Optional[str] = None
    unit: Optional[str] = None
    location: Optional[str] = Field(default=None, description="page or section label of the evidence")
    evidence: Optional[str] = Field(default=None, description="short quote from the reference content")
    reason: str = ""

    @field_validator("baseline", "target", "unit", "document_indicator_text", "location", "evidence", mode="before")
    @classmethod
    def _numbers_to_text(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        return s if s and s.lower() not in ("null", "none", "n/a") else None

    @field_validator("match_type", "level", "disclosure", mode="before")
    @classmethod
    def _lower(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            if v in ("", "null", "none"):
                return None
        return v

    @model_validator(mode="after")
    def _consistent(self):
        if self.disclosure == "absent":
            if self.match_type is not None:
                raise ValueError("match_type must be null when disclosure is 'absent'")
        elif self.match_type is None:
            raise ValueError("match_type (verbatim | close) is required when disclosure is 'present'")
        return self


ANSWER_EXAMPLE = {
    "indikit_id": "5179",
    "disclosure": "present | absent",
    "match_type": "verbatim | close | null",
    "document_indicator_text": "string or null",
    "level": "impact | outcome | output | unclear | null",
    "baseline": "string or null",
    "target": "string or null",
    "unit": "string or null",
    "location": "page/section label",
    "evidence": "short quote from reference content",
    "reason": "one sentence",
}
