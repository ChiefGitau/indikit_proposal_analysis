"""Reference detection (docs/plan.md Section 4.2): statements that the indicators live in another document
("see Annex C - Logical Framework", "Indicator Tracking Table (Annex O)", "cadre logique en annexe", IPTT ...).

A plain annex pointer counts only when the same sentence (or the annex title) mentions indicators / logframe /
tracking; "see Annex 3 for the budget" is not a reference.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

from pipeline import config
from pipeline.preprocess.models import Block, Reference, Table

_INDICATOR_TERMS = r"(?:indicat(?:or|eur)s?|log-?\s?frame|logical\s+framework|results?\s+framework|cadre\s+logique|cadre\s+de\s+r[ée]sultats|performance\s+(?:monitoring|indicator)|m&?e\s+plan|meal\s+plan|iptt|tracking\s+table|indicator\s+(?:tracker|database|matrix|table)|suivi\s+des\s+indicateurs|tableau\s+des\s+indicateurs)"
_ANNEX = r"(?:annexe?|appendix|attachment|pi[èe]ce\s+jointe)\s*(?:n[o°]\.?\s*)?[A-Z0-9]{1,3}(?:[-.][0-9]{1,2})?"
_POINTER = r"(?:see|refer\s+to|cf\.?|detailed\s+in|provided\s+in|included\s+in|attached\s+(?:as|in)|presented\s+in|fill(?:ed)?\s+in|outlined\s+in|voir|cf|se\s+r[ée]f[ée]rer\s+[àa]|pr[ée]sent[ée]s?\s+(?:en|dans)|joint[e]?s?\s+en|ci-joint[e]?s?)"

KIND_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    ("tracker", re.compile(r"\b(?:iptt|indicator\s+tracking\s+table|indicator\s+tracker|indicator\s+performance\s+tracking\s+table|performance\s+indicator\s+tracking\s+table|tableau\s+de\s+suivi\s+des\s+indicateurs)\b", re.IGNORECASE), 0.9),
    ("indicator_table", re.compile(r"\b(?:indicator\s+(?:database|matrix|table|reference\s+sheets?)|performance\s+monitoring\s+plan|\bpmp\b|m&?e\s+plan|meal\s+plan|base\s+de\s+donn[ée]es\s+des\s+indicateurs|tableau\s+des\s+indicateurs|plan\s+de\s+suivi)\b", re.IGNORECASE), 0.8),
    ("logframe", re.compile(r"\b(?:log-?\s?frame|logical\s+framework|results?\s+framework|cadre\s+logique|cadre\s+de\s+r[ée]sultats)\b", re.IGNORECASE), 0.8),
    ("indicator_table", re.compile(r"\bindicat(?:or|eur)s?\b", re.IGNORECASE), 0.6),
]
# "<pointer> Annex X ... <indicator term>" or "<indicator term> ... (Annex X)" within one sentence
_ANNEX_WITH_TERM = re.compile(rf"(?:{_POINTER}\s+(?:the\s+)?(?P<a1>{_ANNEX})[^.;\n]{{0,120}}?{_INDICATOR_TERMS})|(?:{_INDICATOR_TERMS}[^.;\n]{{0,120}}?(?P<a2>{_ANNEX}))", re.IGNORECASE)
_ANNEX_TITLE = re.compile(rf"^\s*(?P<annex>{_ANNEX})\s*[:\-–—.]\s*(?P<title>[^\n]{{3,120}})$", re.IGNORECASE)
_ATTACHED = re.compile(rf"{_INDICATOR_TERMS}[^.;\n]{{0,60}}?\b(?:is|are|will\s+be|est|sont|sera)\s+(?:attached|annexed|enclosed|provided\s+separately|submitted\s+separately|joint[e]?s?|annex[ée]e?s?|en\s+annexe)", re.IGNORECASE)
_IN_ANNEX = re.compile(rf"{_INDICATOR_TERMS}\s+(?:en\s+annexe|in\s+(?:the\s+)?annex(?:es)?)\b", re.IGNORECASE)

# Names of separate indicator documents that count as a reference even without "see Annex" wording
_STANDALONE_DOC = re.compile(r"\b(?:indicator\s+(?:database|tracker|reference\s+sheets?)|base\s+de\s+donn[ée]es\s+des\s+indicateurs|indicator\s+performance\s+tracking\s+table)\b", re.IGNORECASE)

_SENTENCE_SPLIT = re.compile(r"(?<=[.;!?])\s+(?=[A-ZÀ-Ý(\"'])|\n")


def _sentences(text: str) -> Iterable[str]:
    for s in _SENTENCE_SPLIT.split(text or ""):
        s = s.strip()
        if s:
            yield s


def _kind_of(text: str) -> tuple[str, float]:
    for kind, pat, conf in KIND_PATTERNS:
        if pat.search(text):
            return kind, conf
    return "other", 0.6


def _clean_doc_name(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip(" .;:,-–—()")
    return s[:120]


def detect_references_in_text(text: str, location: str) -> list[Reference]:
    refs: list[Reference] = []
    for sent in _sentences(text):
        m_title = _ANNEX_TITLE.match(sent)
        if m_title and re.search(_INDICATOR_TERMS, m_title.group("title"), re.IGNORECASE):
            kind, conf = _kind_of(m_title.group("title"))
            refs.append(Reference(referenced_document=_clean_doc_name(f"{m_title.group('annex')}: {m_title.group('title')}"), quote=sent[:300], location=location, kind=kind, confidence=min(0.95, conf + 0.05)))
            continue
        m = _ANNEX_WITH_TERM.search(sent)
        if m:
            annex = m.group("a1") or m.group("a2")
            kind, conf = _kind_of(sent)
            term = re.search(_INDICATOR_TERMS, sent, re.IGNORECASE)
            name = f"{annex} ({term.group(0)})" if term else annex
            refs.append(Reference(referenced_document=_clean_doc_name(name), quote=sent[:300], location=location, kind=kind, confidence=conf))
            continue
        if _ATTACHED.search(sent) or _IN_ANNEX.search(sent):
            kind, conf = _kind_of(sent)
            term = re.search(_INDICATOR_TERMS, sent, re.IGNORECASE)
            refs.append(Reference(referenced_document=_clean_doc_name(term.group(0) if term else "attached document"), quote=sent[:300], location=location, kind=kind, confidence=conf - 0.1))
            continue
        tracker = KIND_PATTERNS[0][1].search(sent)
        if tracker:
            refs.append(Reference(referenced_document=_clean_doc_name(tracker.group(0)), quote=sent[:300], location=location, kind="tracker", confidence=0.7))
            continue
        standalone = _STANDALONE_DOC.search(sent)
        if standalone:
            refs.append(Reference(referenced_document=_clean_doc_name(standalone.group(0)), quote=sent[:300], location=location, kind="indicator_table", confidence=0.6))
    return refs


def detect_references(blocks: list[Block], tables: Optional[list[Table]] = None) -> list[Reference]:
    """References found in the blocks (and table cells, lower confidence). De-duplicated by (document, kind)."""
    found: list[Reference] = []
    for b in blocks:
        if b.kind == "table":
            continue
        loc = f"p.{b.page}" if b.page else (" > ".join(b.heading_path[-2:]) or "document")
        found.extend(detect_references_in_text(b.text, loc))
    for t in tables or []:
        loc = f"table {t.table_id.rsplit(':', 1)[-1]}" + (f" p.{t.page_start}" if t.page_start else "")
        for row in t.rows:
            for cell in row:
                if len(cell) <= 600:
                    for r in detect_references_in_text(cell, loc):
                        r.confidence = round(max(0.3, r.confidence - 0.2), 2)
                        found.append(r)
    dedup: dict[tuple[str, str], Reference] = {}
    for r in found:
        key = (r.referenced_document.lower(), r.kind)
        if key not in dedup or r.confidence > dedup[key].confidence:
            dedup[key] = r
    return sorted(dedup.values(), key=lambda r: (-r.confidence, r.referenced_document))


def write_references(refs: list[Reference], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.model_dump(mode="json") for r in refs], ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def load_references(doc_id: str, kb_dir: Path = config.KB_DIR) -> list[Reference]:
    """Per-document references (Phase 3 interface)."""
    path = Path(kb_dir) / doc_id / "references.json"
    if not path.exists():
        raise FileNotFoundError(f"no references.json for {doc_id} under {path.parent}; run build-kb")
    return [Reference.model_validate(r) for r in json.loads(path.read_text(encoding="utf-8"))]


def elsewhere_flags(refs: list[Reference], n_logframe_tables: int, n_logframe_sections: int) -> tuple[bool, bool]:
    """(indicators_elsewhere, indicators_also_elsewhere) per docs/phase_2.md Task 3."""
    has_local = (n_logframe_tables + n_logframe_sections) > 0
    has_refs = any(r.kind in ("logframe", "indicator_table", "tracker") for r in refs)
    return (has_refs and not has_local), (has_refs and has_local)
