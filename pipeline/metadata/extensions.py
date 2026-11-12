"""Metadata extensions (paper Section 3.2.2): ``<Knowledge>`` and ``<SearchTerm>`` per indicator.

Three layers, applied in this order so that later layers win:

1. **Deterministic baseline** (``apply_baseline``): no API, always runs.
   ``search_terms`` = indicator name, key phrases from the EN/FR wording (template placeholders
   such as ``[specify the target group]`` removed), known abbreviations found in the texts, sector and
   sub-sector names; lower-cased, de-duplicated, at most 15. ``knowledge`` = purpose + first sentences of
   "how to collect" + "important comments", truncated to about 120 words.
2. **Optional LLM enrichment** (``enrich_records``, CLI flag ``--enrich``): one call per indicator that
   rewrites ``knowledge`` as (a) what the indicator measures, (b) what does *not* count, and proposes extra
   search terms. Cached under ``outputs/cache/metadata/`` keyed by (ID, prompt version, model).
3. **Manual overrides** (``apply_overrides``, ``overrides.yaml``): ID -> ``knowledge``,
   ``search_terms_add``, ``search_terms_remove``; each entry's ``name`` must match the built record or the
   entry is skipped and reported.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field, ValidationError

from pipeline import config
from pipeline.metadata.build_metadata import clean_text, normalise_for_compare
from pipeline.metadata.models import IndicatorMetadata

log = logging.getLogger(__name__)

MAX_BASELINE_TERMS = 15
KNOWLEDGE_MAX_WORDS = 120

# Abbreviations common in ZOA / donor logframes. Matched case-sensitively on word boundaries.
ABBREVIATIONS: tuple[str, ...] = (
    "FCS", "rCSI", "CSI", "HHS", "FIES", "MDD-W", "MDDW", "HDDS", "IDDS", "MAD", "MUAC", "GAM", "SAM", "MAM",
    "VSLA", "SILC", "PIP", "NFI", "CVA", "MPCA", "WASH", "GBV", "SGBV", "IWRM", "DRR", "CSA", "NRM", "IGA",
    "MSME", "SME", "TVET", "JMP", "LCSI", "ODF", "CLTS", "HWTS", "FRC", "IPTT", "MHM", "MHPSS", "PSEA", "AAP",
    "KAP", "CCA", "SRHR", "GDI", "GESI", "CHW", "ECD", "ICT",
)

_STOPWORDS_EN = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "and", "or", "as", "is", "are", "was",
    "were", "be", "been", "that", "which", "who", "whom", "whose", "their", "its", "his", "her", "from", "during",
    "per", "than", "this", "these", "those", "have", "has", "had", "not", "no", "within", "into", "over", "under",
    "through", "among", "each", "any", "all", "least", "more", "most", "out", "up", "it", "they", "them",
    "specify", "specified", "target", "group", "period", "time", "frame", "e.g.", "i.e.", "etc", "such",
}
_STOPWORDS_FR = {
    "le", "la", "les", "l", "un", "une", "des", "de", "du", "d", "et", "ou", "à", "a", "au", "aux", "en", "dans",
    "par", "pour", "sur", "avec", "sans", "qui", "que", "dont", "leur", "leurs", "son", "sa", "ses", "ce", "cette",
    "ces", "est", "sont", "été", "être", "ont", "ayant", "étant", "pas", "ne", "plus", "moins", "préciser",
    "précisez", "spécifiez", "spécifier", "cible", "période", "groupe", "ex", "etc",
}

_PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]|\([^)]*(?:specify|préciser|spécifiez)[^)]*\)", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


# --------------------------------------------------------------------------- #
# Deterministic baseline
# --------------------------------------------------------------------------- #
def strip_placeholders(text: str) -> str:
    """Remove template placeholders such as ``[specify the target group]`` and tidy the result."""
    s = _PLACEHOLDER_RE.sub(" ", text or "")
    s = re.sub(r"\s+", " ", s).strip(" ,;:-")
    return s


def key_phrases(text: str, stopwords: set[str], max_phrases: int = 6, min_len: int = 4) -> list[str]:
    """Runs of non-stopword tokens between stopwords / punctuation, e.g.
    ``'% of households with handwashing facilities with soap and water at home'`` ->
    ``['households', 'handwashing facilities', 'soap', 'water', 'home']``."""
    s = strip_placeholders(text).lower()
    s = re.sub(r"[%#]", " ", s)
    phrases: list[str] = []
    for segment in re.split(r"[,;:()/\"]|\s[-–]\s|\.\s|\.$", s):
        run: list[str] = []
        for tok in re.findall(r"[a-zà-ÿ0-9'’-]+", segment):
            core = tok.strip("'’-")
            if not core or core in stopwords or core.isdigit():
                if run:
                    phrases.append(" ".join(run))
                    run = []
                continue
            run.append(core)
        if run:
            phrases.append(" ".join(run))
    out: list[str] = []
    for p in phrases:
        p = p.strip()
        if len(p) >= min_len and p not in out:
            out.append(p)
        if len(out) >= max_phrases:
            break
    return out


def find_abbreviations(*texts: str) -> list[str]:
    found: list[str] = []
    joined = " ".join(t for t in texts if t)
    for abbr in ABBREVIATIONS:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(abbr)}(?![A-Za-z0-9])", joined) and abbr not in found:
            found.append(abbr)
    return found


def _dedupe_terms(terms: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in terms:
        t = re.sub(r"\s+", " ", (t or "")).strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def baseline_search_terms(rec: IndicatorMetadata, max_terms: int = MAX_BASELINE_TERMS) -> list[str]:
    terms: list[str] = [rec.name]
    terms += key_phrases(rec.kpi_wording_en, _STOPWORDS_EN, max_phrases=6)
    terms += key_phrases(rec.kpi_wording_fr or "", _STOPWORDS_FR, max_phrases=3)
    terms += find_abbreviations(rec.name, rec.kpi_wording_en, rec.purpose or "", rec.how_to_collect or "")
    for a in rec.aspect:
        if a.sector:
            terms.append(a.sector)
        if a.sub_sector:
            terms.append(a.sub_sector)
    return _dedupe_terms(terms)[:max_terms]


def first_sentences(text: str, n: int) -> str:
    text = clean_text(text)
    if not text:
        return ""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return " ".join(sentences[:n]).strip()


def truncate_words(text: str, max_words: int = KNOWLEDGE_MAX_WORDS) -> str:
    words = clean_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(",;:") + " …"


def baseline_knowledge(rec: IndicatorMetadata, max_words: int = KNOWLEDGE_MAX_WORDS) -> str:
    parts = [clean_text(rec.purpose), first_sentences(rec.how_to_collect or "", 2), first_sentences(rec.important_comments or "", 1)]
    text = " ".join(p for p in parts if p)
    if not text and rec.kpi_wording_en:
        text = f"Measures: {strip_placeholders(rec.kpi_wording_en)}."
    return truncate_words(text, max_words)


def apply_baseline(records: list[IndicatorMetadata]) -> list[tuple[str, str]]:
    """Fill ``search_terms`` / ``knowledge`` deterministically. Returns [(id, problem)] for records that still
    have fewer than 3 search terms or empty knowledge."""
    exceptions: list[tuple[str, str]] = []
    for rec in records:
        rec.search_terms = baseline_search_terms(rec)
        rec.knowledge = baseline_knowledge(rec) or None
        rec.field_sources["search_terms"] = "derived"
        if rec.knowledge:
            rec.field_sources["knowledge"] = "derived"
        else:
            rec.field_sources.pop("knowledge", None)
        if len(rec.search_terms) < 3:
            exceptions.append((rec.indikit_id, f"only {len(rec.search_terms)} search term(s)"))
        if not rec.knowledge:
            exceptions.append((rec.indikit_id, "empty knowledge"))
    return exceptions


# --------------------------------------------------------------------------- #
# Optional LLM enrichment
# --------------------------------------------------------------------------- #
class EnrichmentResponse(BaseModel):
    """What the model must return (validated before use)."""

    knowledge: str = Field(min_length=1)
    search_terms: list[str] = Field(default_factory=list)


def build_enrichment_prompt(rec: IndicatorMetadata) -> str:
    return (
        "You are a MEAL specialist maintaining the ZOA IndiKit indicator library. For the indicator below, write\n"
        "(a) one or two sentences on what it measures and (b) one or two sentences on what does NOT count as the\n"
        "same indicator (common look-alikes, activity counts, different units). Then propose up to 10 additional\n"
        "search terms: synonyms, donor phrasings (BHA, EU, UN OCHA codes), French terms, abbreviations.\n"
        "Respond with JSON only: {\"knowledge\": \"<(a) + (b)>\", \"search_terms\": [\"...\"]}.\n\n"
        f"ID: {rec.indikit_id}\nName: {rec.name}\nWording (EN): {rec.kpi_wording_en}\nWording (FR): {rec.kpi_wording_fr or ''}\n"
        f"Sector: {rec.primary_sector} / {rec.primary_sub_sector}\nLevels: {', '.join(rec.levels)}\n"
        f"Purpose: {rec.purpose or ''}\nHow to collect (excerpt): {first_sentences(rec.how_to_collect or '', 3)}\n"
        f"Important comments (excerpt): {first_sentences(rec.important_comments or '', 2)}\n"
        f"Current search terms: {', '.join(rec.search_terms)}\n"
    )


def _cache_path(cache_dir: Path, uid: str, prompt_version: str, model_name: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    return Path(cache_dir) / f"{uid}__{prompt_version}__{safe_model}.json"


def enrich_record(
    rec: IndicatorMetadata,
    model_name: str,
    cache_dir: Path = config.METADATA_CACHE_DIR,
    prompt_version: str = config.METADATA_ENRICH_PROMPT_VERSION,
    call_fn: Optional[Callable[..., str]] = None,
) -> str:
    """Enrich one record in place. Returns "cached", "called" or "invalid". ``call_fn`` defaults to the shared
    retrying client in ``pipeline.agent.answer`` (tests inject a fake)."""
    from pipeline.agent import answer as _answer

    cache_dir = Path(cache_dir)
    path = _cache_path(cache_dir, rec.indikit_id, prompt_version, model_name)
    raw: Optional[dict] = None
    status = "cached"
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8")).get("response")
        except Exception:
            raw = None
    if raw is None:
        status = "called"
        call = call_fn or _answer._call_openai_with_retries
        text = call(build_enrichment_prompt(rec), model_name, response_format={"type": "json_object"}, temperature=0.0, max_tokens=600)
        raw = _answer._extract_json_from_response(text)
        if isinstance(raw, dict):
            cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"indikit_id": rec.indikit_id, "prompt_version": prompt_version, "model": model_name, "response": raw}, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        parsed = EnrichmentResponse.model_validate(raw)
    except ValidationError as exc:
        log.warning("enrichment for %s rejected: %s", rec.indikit_id, exc.errors()[0].get("msg") if exc.errors() else exc)
        return "invalid"
    rec.knowledge = truncate_words(parsed.knowledge, KNOWLEDGE_MAX_WORDS * 2)
    rec.search_terms = _dedupe_terms(rec.search_terms + parsed.search_terms)
    rec.field_sources["knowledge"] = "llm"
    rec.field_sources["search_terms"] = "llm"
    return status


def enrich_records(
    records: list[IndicatorMetadata],
    model_name: Optional[str] = None,
    cache_dir: Path = config.METADATA_CACHE_DIR,
    prompt_version: str = config.METADATA_ENRICH_PROMPT_VERSION,
    call_fn: Optional[Callable[..., str]] = None,
) -> dict[str, int]:
    """Enrich every record (explicit opt-in; never called by the deterministic build)."""
    from pipeline.agent import answer as _answer

    model_name = model_name or _answer.MODEL_NAME
    counts = {"cached": 0, "called": 0, "invalid": 0, "failed": 0, "model": model_name}  # type: ignore[dict-item]
    for rec in records:
        try:
            counts[enrich_record(rec, model_name, cache_dir, prompt_version, call_fn)] += 1
        except Exception as exc:  # one failing indicator must not abort the build
            log.warning("enrichment failed for %s: %s", rec.indikit_id, exc)
            counts["failed"] += 1
    return counts


# --------------------------------------------------------------------------- #
# Manual overrides
# --------------------------------------------------------------------------- #
class Override(BaseModel):
    name: Optional[str] = None
    knowledge: Optional[str] = None
    search_terms_add: list[str] = Field(default_factory=list)
    search_terms_remove: list[str] = Field(default_factory=list)


def load_overrides(path: Path = config.OVERRIDES_YAML) -> dict[str, Override]:
    path = Path(path)
    if not path.exists():
        return {}
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of indikit_id -> override")
    return {str(k): Override.model_validate(v or {}) for k, v in data.items()}


def names_match(expected: Optional[str], actual: str, threshold: int = 85) -> bool:
    """Exact after normalisation, else a token-set similarity (rapidfuzz) of at least ``threshold``."""
    if not expected:
        return True
    a, b = normalise_for_compare(expected), normalise_for_compare(actual)
    if a == b:
        return True
    try:
        from rapidfuzz import fuzz
    except Exception:  # pragma: no cover
        return False
    return fuzz.token_set_ratio(a, b) >= threshold


def apply_overrides(records: list[IndicatorMetadata], overrides: dict[str, Override]) -> tuple[list[str], list[tuple[str, str]]]:
    """Apply overrides last (they always win). Returns (applied_ids, [(id, reason) skipped])."""
    by_id = {r.indikit_id: r for r in records}
    applied: list[str] = []
    skipped: list[tuple[str, str]] = []
    for uid, ov in sorted(overrides.items()):
        rec = by_id.get(uid)
        if rec is None:
            skipped.append((uid, "ID not in metadata"))
            continue
        if not names_match(ov.name, rec.name):
            skipped.append((uid, f"name mismatch: override '{ov.name}' vs built '{rec.name}'"))
            continue
        if ov.knowledge:
            rec.knowledge = clean_text(ov.knowledge)
            rec.field_sources["knowledge"] = "override"
        remove = {t.lower().strip() for t in ov.search_terms_remove}
        terms = [t for t in rec.search_terms if t not in remove]
        rec.search_terms = _dedupe_terms(terms + list(ov.search_terms_add))
        if ov.search_terms_add or ov.search_terms_remove:
            rec.field_sources["search_terms"] = "override"
        applied.append(uid)
    return applied, skipped
