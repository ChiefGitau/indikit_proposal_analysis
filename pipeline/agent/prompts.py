"""Prompt templates - the metadata module's *Expressions* (paper Section 3.2) rendered as the five-part
prompt of the LLM agent (paper Section 3.4.2):

1. preset information (role and rules),
2. reference content (retrieved passages / table rows with location labels),
3. expert knowledge (the indicator's Knowledge field + the match rubric),
4. the question,
5. the answer format (JSON, validated with ``DisclosureAnswer``).

Phase 1 ships the template only; nothing here calls a model. Bump ``config.PROMPT_VERSION`` when the
wording changes so cached answers are not reused.
"""
from __future__ import annotations

import json
from typing import Iterable

from pipeline import config
from pipeline.agent.models import ANSWER_EXAMPLE, DisclosureAnswer
from pipeline.metadata.models import IndicatorMetadata

PROMPT_VERSION = config.PROMPT_VERSION

PART_HEADERS = (
    "## 1. Preset information",
    "## 2. Reference content",
    "## 3. Expert knowledge",
    "## 4. Question",
    "## 5. Answer format",
)

PRESET_INFORMATION = """You are a monitoring, evaluation, accountability and learning (MEAL) specialist at ZOA.
You check whether a project document (proposal, logframe, contract) contains a given ZOA IndiKit indicator.

Rules:
- Answer only from the reference content below. If the evidence is not there, answer "absent".
- An activity (training held, items distributed, meetings organised) is not an indicator unless it is phrased as a measured result with a target.
- One table row may hold several indicators; judge only the one that matches the indicator described.
- Keep donor codes (e.g. W8, S4, K1) and the document's exact wording in "document_indicator_text".
- Copy baseline and target as written in the document (keep units and symbols); use null when not stated.
- Never invent values, locations or quotes."""

MATCH_RUBRIC = """Match types (docs/plan.md Section 1.3):
- verbatim: same measurement and (nearly) identical wording, e.g. "Number of people having access to basic, safe and dignified shelter solutions" <-> IndiKit 5179 Access to Adequate Shelter.
- close: same measurement, different wording or scope, e.g. "75% of trained PIP farmers report increased yields" <-> IndiKit 5260 Agricultural Productivity.
- absent: no indicator in the document measures the same thing (activity counts such as trainings held or VSLAs supported do not count)."""


def _indicator_block(ind: IndicatorMetadata) -> str:
    aspects = "; ".join(f"{a.sector}" + (f" > {a.sub_sector}" if a.sub_sector else "") for a in ind.aspect) or "(unknown)"
    lines = [
        f"IndiKit ID: {ind.indikit_id}",
        f"Name: {ind.name}",
        f"Wording (EN): {ind.kpi_wording_en or '(not available)'}",
    ]
    if ind.kpi_wording_fr:
        lines.append(f"Wording (FR): {ind.kpi_wording_fr}")
    lines += [
        f"Aspect (sector > sub-sector): {aspects}",
        f"Levels: {', '.join(ind.levels) or '(unknown)'}",
        f"Quantity: {ind.quantity.value}",
        f"Topic (disaggregation): {ind.topic or '(none)'}",
        f"Search terms: {', '.join(ind.search_terms) or '(none)'}",
    ]
    return "\n".join(lines)


def format_reference_content(reference_content: Iterable[str]) -> str:
    items = [c.strip() for c in reference_content if c and c.strip()]
    if not items:
        return "(no reference content retrieved)"
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(items, start=1))


def answer_format_block() -> str:
    schema = DisclosureAnswer.model_json_schema()
    return (
        "Respond with a single JSON object and nothing else, exactly in this shape:\n"
        + json.dumps(ANSWER_EXAMPLE, indent=2, ensure_ascii=False)
        + "\n\nJSON schema (for validation):\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def build_question(ind: IndicatorMetadata) -> str:
    aspect = ind.primary_sector + (f" / {ind.primary_sub_sector}" if ind.primary_sub_sector else "")
    kpi = ind.kpi_wording_en or ind.name
    topic = f", disaggregated by {ind.topic}" if ind.topic else ""
    return (
        f"Does this document contain an indicator that measures \"{kpi}\" for {aspect or 'the relevant sector'}{topic}? "
        "If so, give its exact wording, result level, baseline, target, unit and location, and say whether the match is verbatim or close. "
        "If not, answer \"absent\"."
    )


def build_disclosure_prompt(indicator: IndicatorMetadata, reference_content: list[str], rubric: str = MATCH_RUBRIC) -> str:
    """Assemble the five-part prompt for one (document, indicator) pair."""
    knowledge = indicator.knowledge or "(no knowledge note available)"
    parts = [
        f"{PART_HEADERS[0]}\n{PRESET_INFORMATION}\n\nIndicator under review:\n{_indicator_block(indicator)}",
        f"{PART_HEADERS[1]}\n{format_reference_content(reference_content)}",
        f"{PART_HEADERS[2]}\nKnowledge note for this indicator:\n{knowledge}\n\n{rubric}",
        f"{PART_HEADERS[3]}\n{build_question(indicator)}",
        f"{PART_HEADERS[4]}\n{answer_format_block()}",
    ]
    return f"<!-- prompt version {PROMPT_VERSION} -->\n" + "\n\n".join(parts) + "\n"
