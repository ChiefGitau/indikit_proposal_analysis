"""Prompt templates - the metadata module's *Expressions* (paper Section 3.2) rendered as the five-part
prompt of the LLM agent (paper Section 3.4.2):

1. preset information (role and rules),
2. reference content (retrieved passages / table rows, each labelled ``[E1] (page 12, Logical Framework > Outcome 2)``),
3. expert knowledge (the indicator's Knowledge field + the match rubric; switchable for the ablation),
4. the question, built from the metadata entity <Aspect, KPI, Topic, Quantity>,
5. the answer format (JSON, validated with ``DisclosureAnswer``).

Part 1 is sent as the system message, parts 2-5 as the user message. Bump ``config.PROMPT_VERSION`` when the
wording changes so cached answers are not reused. Also holds the novel-pass extraction prompt and the
reverse-check prompt (docs/phase_3.md Task 5).
"""
from __future__ import annotations

import json
from typing import Iterable, Optional, Sequence

from pipeline import config
from pipeline.agent.models import ANSWER_EXAMPLE, DisclosureAnswer, ExtractionAnswer, ReverseCheckAnswer
from pipeline.agent.rubric import load_rubric
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
- Answer only from the reference content. Do not use outside knowledge about the project or the country.
- An activity (training held, items distributed, meetings organised, groups supported) is not an indicator unless it is phrased as a measured result.
- One table row may hold several indicators; judge each one separately and report every one that matches.
- Copy "document_indicator_text" and "evidence_quote" exactly as they appear in the reference content (same words, same order), and name the evidence item they come from.
- Copy baseline and target as written in the document (keep units and symbols); use null when not stated.
- If you are unsure whether the same thing is measured, answer "absent". Never invent values, locations or quotes."""

# Backwards-compatible name used by Phase 1 code / tests
MATCH_RUBRIC = load_rubric()


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
    ]
    return "\n".join(lines)


def evidence_label(i: int) -> str:
    return f"E{i}"


def format_reference_items(items: Sequence[tuple[str, str]]) -> str:
    """``items`` = (location, text) in rank order -> ``[E1] (location)\\ntext``."""
    items = [(loc, (txt or "").strip()) for loc, txt in items if txt and txt.strip()]
    if not items:
        return "(no reference content retrieved)"
    return "\n\n".join(f"[{evidence_label(i)}] ({loc or 'location unknown'})\n{txt}" for i, (loc, txt) in enumerate(items, start=1))


def format_reference_content(reference_content: Iterable[str]) -> str:
    """Plain strings (Phase 1 interface): labelled E1.. without a location."""
    return format_reference_items([("", c) for c in reference_content])


def evidence_items(evidence) -> list[tuple[str, str]]:
    """(location label, display text) per ``Evidence`` object, e.g. ``page 12, Logical Framework > Outcome 2``."""
    out = []
    for e in evidence:
        loc = (e.location or "").replace("p.", "page ")
        path = ""
        if e.kb_type == "table":
            first = (e.display_text or "").split("\n", 1)[0]
            if first.startswith("Table: "):
                path = first[len("Table: "):].split(" [", 1)[0]
        label = ", ".join(x for x in (loc, path) if x) or "location unknown"
        out.append((label, e.display_text))
    return out


def answer_format_block() -> str:
    return (
        "Respond with a single JSON object and nothing else, exactly in this shape:\n"
        + json.dumps(ANSWER_EXAMPLE, indent=2, ensure_ascii=False)
        + "\n\nRules: \"disclosure\" is \"present\" when \"matches\" has at least one item and \"absent\" when it is empty. "
        "\"evidence_label\" must be one of the labels in the reference content. Keep \"evidence_quote\" short (one row or one sentence)."
        + "\n\nJSON schema (for validation):\n"
        + json.dumps(DisclosureAnswer.model_json_schema(), ensure_ascii=False)
    )


def build_question(ind: IndicatorMetadata) -> str:
    aspect = ind.primary_sector + (f" / {ind.primary_sub_sector}" if ind.primary_sub_sector else "")
    kpi = ind.kpi_wording_en or ind.name
    topic = f", disaggregated by {ind.topic}" if ind.topic else ""
    return (
        f"Does the reference content contain one or more indicators that measure \"{kpi}\" ({ind.name}; {ind.quantity.value}) "
        f"for {aspect or 'the relevant sector'}{topic}? For each, give the exact wording and its values (level, baseline, target, unit), "
        "the evidence label and a short exact quote, and say whether the match is verbatim or close. If none, answer \"absent\" with an empty list."
    )


def build_disclosure_parts(indicator: IndicatorMetadata, reference_items: Sequence[tuple[str, str]], *, use_knowledge: bool = True, rubric: Optional[str] = None) -> list[str]:
    rubric = load_rubric() if rubric is None else rubric
    knowledge = indicator.knowledge or "(no knowledge note available)"
    expert = (f"Knowledge note for this indicator:\n{knowledge}\n\n{rubric}" if use_knowledge else f"{rubric}")
    return [
        f"{PART_HEADERS[0]}\n{PRESET_INFORMATION}\n\nIndicator under review:\n{_indicator_block(indicator)}",
        f"{PART_HEADERS[1]}\n{format_reference_items(reference_items)}",
        f"{PART_HEADERS[2]}\n{expert}",
        f"{PART_HEADERS[3]}\n{build_question(indicator)}",
        f"{PART_HEADERS[4]}\n{answer_format_block()}",
    ]


def build_disclosure_messages(indicator: IndicatorMetadata, reference_items: Sequence[tuple[str, str]], *, use_knowledge: bool = True, rubric: Optional[str] = None) -> tuple[str, str]:
    """(system, user): part 1 is the system message; parts 2-5 the user message."""
    parts = build_disclosure_parts(indicator, reference_items, use_knowledge=use_knowledge, rubric=rubric)
    header = f"<!-- prompt version {PROMPT_VERSION} | rubric {config.RUBRIC_VERSION} | knowledge {'on' if use_knowledge else 'off'} -->"
    return parts[0], header + "\n" + "\n\n".join(parts[1:]) + "\n"


def build_disclosure_prompt(indicator: IndicatorMetadata, reference_content, rubric: Optional[str] = None, *, use_knowledge: bool = True) -> str:
    """Assemble the five-part prompt as one string (system + user). ``reference_content`` is a list of strings
    (Phase 1 interface) or of ``(location, text)`` tuples."""
    items = [(c if isinstance(c, tuple) else ("", c)) for c in reference_content]
    system, user = build_disclosure_messages(indicator, items, use_knowledge=use_knowledge, rubric=rubric)
    return system + "\n\n" + user


# --------------------------------------------------------------------------- #
# Novel-indicator pass: extraction from logframe text sections and unmapped tables (Task 5A)
# --------------------------------------------------------------------------- #
EXTRACTION_SYSTEM = """You are a MEAL specialist at ZOA. You extract monitoring indicators from a section of a project document.
Rules:
- Work only from the text given. Copy each indicator's wording exactly; do not paraphrase or merge indicators.
- An indicator measures a result (a number, percentage, score, ratio or yes/no). Activities, deliverables, assumptions, risks and budget lines are not indicators: list them with "is_indicator": false (or leave them out).
- When a line holds several indicators, split them into separate items.
- Copy baseline and target as written; use null when not stated. Level is one of impact, outcome, output, unclear.
- "source_quote" is the exact text the indicator was taken from."""

EXTRACTION_EXAMPLE = {"indicators": [{"indicator_text": "exact wording", "is_indicator": True, "level": "outcome | output | impact | unclear", "baseline": "string or null", "target": "string or null", "unit": "string or null", "means_of_verification": "string or null", "result_context": "string or null", "source_quote": "exact text"}]}


def build_extraction_messages(section_text: str, *, kind: str = "text", title: str = "", rubric: Optional[str] = None) -> tuple[str, str]:
    rubric = load_rubric() if rubric is None else rubric
    what = "logframe table whose columns could not be identified" if kind == "table" else "logframe section"
    user = (
        f"<!-- prompt version {config.NOVEL_PROMPT_VERSION} | rubric {config.RUBRIC_VERSION} -->\n"
        f"## Section\n{what}{(': ' + title) if title else ''}\n\n{section_text.strip()}\n\n"
        f"## Rubric (what is and is not an indicator)\n{rubric}\n\n"
        "## Task\nList every indicator in the section with its values. Return {\"indicators\": []} if there is none.\n\n"
        "## Answer format\nRespond with a single JSON object exactly in this shape:\n" + json.dumps(EXTRACTION_EXAMPLE, indent=2, ensure_ascii=False)
        + "\n\nJSON schema (for validation):\n" + json.dumps(ExtractionAnswer.model_json_schema(), ensure_ascii=False) + "\n"
    )
    return EXTRACTION_SYSTEM, user


# --------------------------------------------------------------------------- #
# Reverse check (Task 5C)
# --------------------------------------------------------------------------- #
REVERSE_SYSTEM = """You are a MEAL specialist at ZOA. You decide whether a document indicator is the same measurement as one of the candidate ZOA IndiKit indicators.
Rules:
- Use the rubric. A match must measure the same thing (same numerator and population kind), not merely the same topic.
- It is normal that none of the candidates matches: then return an empty list.
- Report every candidate that matches (verbatim or close), nothing else."""

REVERSE_EXAMPLE = {"matches": [{"indikit_id": "5138", "match_type": "verbatim | close", "reason": "one sentence"}], "reason": "one sentence (also when no candidate matches)"}


def build_reverse_check_messages(doc_indicator_text: str, candidates: Sequence[IndicatorMetadata], *, context: str = "", rubric: Optional[str] = None) -> tuple[str, str]:
    rubric = load_rubric() if rubric is None else rubric
    cands = []
    for c in candidates:
        block = [f"- IndiKit ID {c.indikit_id}: {c.name}", f"  Wording: {c.kpi_wording_en or '(not available)'}"]
        if c.knowledge:
            block.append(f"  Knowledge: {c.knowledge}")
        cands.append("\n".join(block))
    user = (
        f"<!-- prompt version {config.REVERSE_CHECK_PROMPT_VERSION} | rubric {config.RUBRIC_VERSION} -->\n"
        f"## Document indicator\n\"{doc_indicator_text.strip()}\"" + (f"\nContext: {context.strip()}" if context else "") + "\n\n"
        f"## Candidate IndiKit indicators\n" + "\n".join(cands) + "\n\n"
        f"## Rubric\n{rubric}\n\n"
        "## Question\nIs the document indicator a verbatim or close match of any candidate? Return all matching candidates, or an empty list.\n\n"
        "## Answer format\nRespond with a single JSON object exactly in this shape:\n" + json.dumps(REVERSE_EXAMPLE, indent=2, ensure_ascii=False)
        + "\n\nJSON schema (for validation):\n" + json.dumps(ReverseCheckAnswer.model_json_schema(), ensure_ascii=False) + "\n"
    )
    return REVERSE_SYSTEM, user
