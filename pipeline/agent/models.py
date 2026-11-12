"""Answer schemas of the LLM agent (paper Section 3.4.2 answer format; docs/phase_3.md Tasks 3 and 5).

The paper's answer is <Disclosure, KPI, Topic, Value, Unit, Target, Action>; ``DisclosureAnswer`` mirrors it with
``match_type`` (verbatim / close) for IndiKit and a list of ``matches`` because one IndiKit indicator can match
several document indicators. ``disclosure = present`` if and only if ``matches`` is non-empty.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Level = Literal["impact", "outcome", "output", "unclear"]
MatchType = Literal["verbatim", "close"]


def _opt_text(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        v = "; ".join(str(x) for x in v if x is not None)
    s = str(v).strip()
    return s if s and s.lower() not in ("null", "none", "n/a", "na", "not stated", "not specified", "-") else None


def _lower_enum(v):
    if isinstance(v, str):
        v = v.strip().lower()
        if v in ("", "null", "none"):
            return None
    return v


class MatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_indicator_text: str = Field(description="exact wording copied from the reference content")
    match_type: MatchType
    level: Level = "unclear"
    baseline: Optional[str] = None
    target: Optional[str] = None
    unit: Optional[str] = None
    evidence_label: str = Field(description="label of the evidence item, e.g. E2")
    evidence_quote: str = Field(default="", description="short exact quote from that evidence")
    reason: str = ""

    @field_validator("baseline", "target", "unit", mode="before")
    @classmethod
    def _opt(cls, v):
        return _opt_text(v)

    @field_validator("match_type", mode="before")
    @classmethod
    def _mt(cls, v):
        return _lower_enum(v)

    @field_validator("level", mode="before")
    @classmethod
    def _lvl(cls, v):
        v = _lower_enum(v)
        if v is None:
            return "unclear"
        from pipeline.models import normalize_indicator_level

        lvl = normalize_indicator_level(v)
        return lvl if lvl in ("impact", "outcome", "output") else "unclear"

    @field_validator("evidence_label", mode="before")
    @classmethod
    def _label(cls, v):
        s = str(v or "").strip().strip("[]").upper()
        return s

    @field_validator("document_indicator_text", "evidence_quote", "reason", mode="before")
    @classmethod
    def _text(cls, v):
        return "" if v is None else str(v).strip()


class DisclosureAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indikit_id: str
    disclosure: Literal["present", "absent"]
    matches: list[MatchItem] = Field(default_factory=list)

    @field_validator("indikit_id", mode="before")
    @classmethod
    def _id(cls, v):
        return str(v).strip()

    @field_validator("disclosure", mode="before")
    @classmethod
    def _disc(cls, v):
        return _lower_enum(v)

    @model_validator(mode="after")
    def _present_iff_matches(self):
        # disclosure = present <=> matches non-empty (coerced rather than rejected: the quote check decides what counts)
        object.__setattr__(self, "disclosure", "present" if self.matches else "absent")
        return self


ANSWER_EXAMPLE = {
    "indikit_id": "5239",
    "disclosure": "present | absent",
    "matches": [
        {
            "document_indicator_text": "exact wording copied from the reference content",
            "match_type": "verbatim | close",
            "level": "impact | outcome | output | unclear",
            "baseline": "string or null",
            "target": "string or null",
            "unit": "string or null",
            "evidence_label": "E2",
            "evidence_quote": "short exact quote from that evidence",
            "reason": "one sentence",
        }
    ],
}


# --------------------------------------------------------------------------- #
# Novel-indicator pass (Task 5A): extraction from logframe text / unmapped tables
# --------------------------------------------------------------------------- #
class ExtractedIndicator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indicator_text: str
    is_indicator: bool = True                  # False for activities / deliverables / assumptions (rubric "not an indicator")
    level: Level = "unclear"
    baseline: Optional[str] = None
    target: Optional[str] = None
    unit: Optional[str] = None
    means_of_verification: Optional[str] = None
    result_context: Optional[str] = None       # the outcome / output the indicator belongs to, if stated
    source_quote: str = ""                     # exact text the indicator was taken from

    @field_validator("baseline", "target", "unit", "means_of_verification", "result_context", mode="before")
    @classmethod
    def _opt(cls, v):
        return _opt_text(v)

    @field_validator("level", mode="before")
    @classmethod
    def _lvl(cls, v):
        v = _lower_enum(v)
        if v is None:
            return "unclear"
        from pipeline.models import normalize_indicator_level

        lvl = normalize_indicator_level(v)
        return lvl if lvl in ("impact", "outcome", "output") else "unclear"

    @field_validator("indicator_text", "source_quote", mode="before")
    @classmethod
    def _text(cls, v):
        return "" if v is None else " ".join(str(v).split())


class ExtractionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indicators: list[ExtractedIndicator] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Reverse check (Task 5C): is this document indicator a verbatim / close match of one of N candidates?
# --------------------------------------------------------------------------- #
class ReverseMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indikit_id: str
    match_type: MatchType
    reason: str = ""

    @field_validator("indikit_id", mode="before")
    @classmethod
    def _id(cls, v):
        return str(v).strip()

    @field_validator("match_type", mode="before")
    @classmethod
    def _mt(cls, v):
        return _lower_enum(v)


class ReverseCheckAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matches: list[ReverseMatch] = Field(default_factory=list)   # empty = none of the candidates matches
    reason: str = ""


# --------------------------------------------------------------------------- #
# Pipeline records (not LLM schemas)
# --------------------------------------------------------------------------- #
class ValidatedMatch(BaseModel):
    """A ``MatchItem`` after post-validation: evidence resolved, quotes checked, values normalised."""

    indikit_id: str
    document_indicator_text: str
    match_type: MatchType
    level: Level = "unclear"
    baseline: Optional[str] = None
    target: Optional[str] = None
    baseline_norm: Optional[str] = None
    target_norm: Optional[str] = None
    unit: Optional[str] = None
    evidence_label: str = ""
    evidence_quote: str = ""
    reason: str = ""
    entry_id: Optional[str] = None
    location: str = ""
    kb_type: str = ""
    table_id: Optional[str] = None
    row_id: Optional[str] = None
    block_ids: list[str] = Field(default_factory=list)
    rerank_score: Optional[float] = None
    vector_score: Optional[float] = None
    supported: bool = True
    unsupported_reason: Optional[str] = None
    text_similarity: Optional[float] = None       # similarity of the document text to the IndiKit wording (0-100)
    source: str = "disclosure"                     # disclosure | reverse_check


class PairAnswer(BaseModel):
    """Result of ``answer_pair``: the validated answer, the evidence it saw, usage and flags."""

    doc_id: str
    indikit_id: str
    disclosure: Literal["present", "absent", "failed"] = "absent"
    matches: list[ValidatedMatch] = Field(default_factory=list)       # supported matches only
    unsupported: list[ValidatedMatch] = Field(default_factory=list)   # dropped by the quote check (counted in the run report)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_fingerprint: str = ""
    best_rerank_score: Optional[float] = None
    best_vector_score: Optional[float] = None
    n_evidence: int = 0
    cached: bool = False
    model: str = ""
    prompt_version: str = ""
    rubric_version: str = ""
    use_knowledge: bool = True
    use_search_terms: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 0
    error: Optional[str] = None
    raw_answer: Optional[dict] = None


class DocIndicator(BaseModel):
    """One indicator found in a document (code-first extraction, LLM extraction or disclosure only)."""

    doc_indicator_id: str
    doc_id: str
    indicator_text: str
    level: Level = "unclear"
    baseline: Optional[str] = None
    target: Optional[str] = None
    unit: Optional[str] = None
    means_of_verification: Optional[str] = None
    result_context: Optional[str] = None
    location: str = ""
    table_id: Optional[str] = None
    row_id: Optional[str] = None
    block_ids: list[str] = Field(default_factory=list)
    source: str = "extraction"      # extraction (table_code | table_llm | text_llm) | disclosure_only
    extraction_method: str = ""     # table_code | table_llm | text_llm | ""
    is_indicator: bool = True
    order: int = 0


class LinkedMatch(BaseModel):
    indikit_id: str
    indikit_name: str = ""
    match_type: MatchType
    source: str = "disclosure"       # disclosure | reverse_check
    rerank_score: Optional[float] = None
    reason: str = ""
    evidence_quote: str = ""
    baseline: Optional[str] = None
    target: Optional[str] = None
    unit: Optional[str] = None
    text_similarity: Optional[float] = None


class ConsolidatedIndicator(BaseModel):
    indicator: DocIndicator
    matches: list[LinkedMatch] = Field(default_factory=list)
    primary: Optional[LinkedMatch] = None
    alternate_ids: list[str] = Field(default_factory=list)
    nearest_indikit: list[tuple[str, float]] = Field(default_factory=list)   # (indikit_id, cosine) for novel candidates

    @property
    def match_type(self) -> str:
        return self.primary.match_type if self.primary else "none"


class DocumentResult(BaseModel):
    """Everything a run produced for one document (persisted as ``outputs/runs/<run_id>/docs/<doc_id>.json``)."""

    doc_id: str
    relative_path: str = ""
    country: str = ""
    region: str = ""
    language: str = ""
    file_type: str = ""
    parse_status: str = ""
    indicators_elsewhere: bool = False
    indicators_also_elsewhere: bool = False
    referenced_documents: list[str] = Field(default_factory=list)
    skipped_reason: Optional[str] = None
    consolidated: list[ConsolidatedIndicator] = Field(default_factory=list)
    pairs_selected: int = 0
    pairs_asked: int = 0
    pairs_cached: int = 0
    pairs_present: int = 0
    pairs_failed: int = 0
    unsupported_dropped: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    novel_stats: dict = Field(default_factory=dict)
    reverse_stats: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    budget_stop: Optional[str] = None
    export_only_matches: int = 0
