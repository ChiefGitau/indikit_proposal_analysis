"""Build the IndiKit metadata library: entities (paper Section 3.2.1) merged from the CSV export
and the site scrape, then extended (Section 3.2.2, see ``extensions``).

Sources (all read-only, locations in ``pipeline.config``):

- the IndiKit export CSV (latin-1 / cp1252; one row per sub-sector assignment, so IDs repeat);
- ``scraped.json`` (303 live indicator pages + the levels read from the sector listings);
- the site CSV from the notebook scrape (section texts, used to fill gaps);
- an optional ``site_supplement.json`` written by ``fetch_missing.py`` (PIP indicators);
- a seed list for the PIP indicators when no supplement exists.

Merge rules (docs/phase_1.md Task 3): the export's repeated rows collapse into the ``aspect`` list;
when export and site disagree on wording, purpose, name or sector the **site wins** and the conflict
is recorded; export-only indicators are kept with ``on_site=False``.

The build is deterministic: same inputs give a byte-identical JSONL.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from pipeline import config
from pipeline.metadata.models import Aspect, IndicatorMetadata, Quantity

log = logging.getLogger(__name__)

EXPORT_ID = "Unique ID"
EXPORT_NAME = "Indicator Name"
EXPORT_WORDING_EN = "Indicator Wording - ENG"
EXPORT_WORDING_FR = "Indicator Wording - FRA"
EXPORT_PURPOSE = "What is its Purpose?"
EXPORT_GUIDANCE = "Guidance"
EXPORT_LEVEL = "Indicator Level"
EXPORT_SECTION, EXPORT_SECTOR, EXPORT_SUB_SECTOR = "Section", "Sector", "Sub-Sector"
EXPORT_REQUIRED = (EXPORT_ID, EXPORT_NAME, EXPORT_WORDING_EN, EXPORT_SECTION, EXPORT_SECTOR)


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
_C1 = range(0x80, 0xA0)
_PUNCT_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", "\xa0": " "})


def fix_mojibake(text: str) -> str:
    """Repair text decoded as latin-1 that was really cp1252 (’ – “ ” … œ appear as C1 control chars)."""
    if not text:
        return ""
    out = []
    for ch in text:
        o = ord(ch)
        if o in _C1:
            try:
                ch = bytes([o]).decode("cp1252")
            except UnicodeDecodeError:
                pass
        out.append(ch)
    return "".join(out).replace("\xa0", " ")


def clean_text(value) -> str:
    """None/NaN -> ''; repair mojibake; collapse whitespace."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    s = fix_mojibake(str(value))
    return re.sub(r"\s+", " ", s).strip()


def normalise_for_compare(value) -> str:
    """Lower-case, unify quotes/dashes, drop punctuation: used to decide whether two values really differ."""
    s = clean_text(value).lower().translate(_PUNCT_MAP)
    s = re.sub(r"[^\w%#]+", " ", s)
    return s.strip()


def split_indikit_wording(wording) -> tuple[str, str]:
    """Split an IndiKit "Indicator's Wording" block into (english, french).

    The block is normally "English\\nFrench\\n<english wording>\\n<french wording>", with occasional
    hard line breaks (\\r\\n) inside a sentence. Returns empty strings when nothing usable is present.
    """
    if not wording or not isinstance(wording, str):
        return "", ""
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in wording.replace("\r\n", " ").split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return "", ""
    if len(lines) >= 3 and lines[0].lower() == "english" and lines[1].lower() == "french":
        return lines[2], " ".join(lines[3:])
    return lines[0], ""


# --------------------------------------------------------------------------- #
# Field derivation rules
# --------------------------------------------------------------------------- #
_LEVEL_CANON = {
    "output": "Output",
    "outputs": "Output",
    "outcome": "Outcome",
    "outcomes": "Outcome",
    "impact": "Impact",
    "organisational level": "Organisational level",
    "organizational level": "Organisational level",
    "organisational": "Organisational level",
    "organizational": "Organisational level",
}


def normalise_levels(raw) -> list[str]:
    """``"Output | Outcome"`` / ``["Outcome"]`` / None -> canonical, ordered, de-duplicated level list."""
    if raw is None:
        items: list = []
    elif isinstance(raw, str):
        items = re.split(r"[|,;/]", raw)
    else:
        items = []
        for x in raw:
            items.extend(normalise_levels(x))
    found = set()
    for item in items:
        key = clean_text(item).lower()
        if not key or key == "(missing)":
            continue
        canon = _LEVEL_CANON.get(key)
        if canon is None:
            log.warning("unknown indicator level %r ignored", item)
            continue
        found.add(canon)
    return [lvl for lvl in config.LEVELS if lvl in found]


_YES_NO_RE = re.compile(
    r"^(whether|existence of|presence of|availability of|yes\s*/\s*no|(organisation|organization|project|agency|platform|plan|policy|system|mechanism)\b.*\b(is|are|has|have|was|were|exists?)\b)"
)
_PERCENT_START_RE = re.compile(r"^(%|(?:percentage|percent|proportion|share|prevalence|incidence)\b)")
_NUMBER_START_RE = re.compile(
    r"^(#|(?:number|no\.|nr\.|total number|total amount|amount|quantity|total|count|volume|value|total value|total usd|usd value|"
    r"hectares|ha|tonnes|tons|kg|kilograms|litres|liters|km|kilometres|kilometers|m2|m3|days|months|hours|weeks)\b)"
)
_AVERAGE_START_RE = re.compile(r"^(average|mean|median)\b")
_SCORE_RE = re.compile(r"\b(score|scores|scored|index|scale|marks)\b")
_LABEL_PREFIX_RE = re.compile(r"^[a-z]{2,6}:\s*")
_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")


def derive_quantity(wording: str, name: str = "") -> Quantity:
    """Rule-based measurement type from the English wording, falling back to the name.

    After dropping a label prefix ("PIP: ") and a leading article, the precedence is:
    yes/no cue > percentage cue at the start > number cue at the start (incl. "#" and unit words) >
    'score' / 'index' / 'scale' in name or wording > 'ratio' > average / mean > percentage anywhere >
    "number of" anywhere > other.
    """
    w = clean_text(wording).lower().lstrip("(").strip()
    n = clean_text(name).lower()
    text = w or n
    text = _ARTICLE_RE.sub("", _LABEL_PREFIX_RE.sub("", text))
    if not text:
        return Quantity.other
    if _YES_NO_RE.match(text) or "yes/no" in text:
        return Quantity.yes_no
    if _PERCENT_START_RE.match(text):
        return Quantity.percentage
    if _NUMBER_START_RE.match(text):
        return Quantity.number
    if _SCORE_RE.search(n) or _SCORE_RE.search(text):
        return Quantity.score
    if re.search(r"\bratio\b", n) or re.search(r"\bratio\b", text):
        return Quantity.ratio
    if _AVERAGE_START_RE.match(text):
        return Quantity.number
    if "%" in text or re.search(r"\b(percentage|proportion|rate|share of|prevalence)\b", text):
        return Quantity.percentage
    if re.search(r"\b(number of|amount of|quantity of)\b|#", text):
        return Quantity.number
    return Quantity.other


def name_from_slug(url: str) -> str:
    """``.../5889-pip-of-households-who-...`` -> ``PIP of households who ...`` (used for seeded records)."""
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    words = slug.split("-")[1:] if slug.split("-", 1)[0].isdigit() else slug.split("-")
    words = ["PIP" if w.lower() == "pip" else w for w in words]
    text = " ".join(words).strip()
    return text[:1].upper() + text[1:] if text else slug


def id_from_url(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    uid = slug.split("-", 1)[0]
    return uid if uid.isdigit() else slug


def _sort_key(uid: str):
    return (0, int(uid)) if uid.isdigit() else (1, uid)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def read_export(path: Path = config.EXPORT_CSV, encoding: str = config.EXPORT_ENCODING) -> pd.DataFrame:
    """Read the export as latin-1, drop the empty trailing columns, clean every cell."""
    df = pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in EXPORT_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"export CSV {path} lacks columns {missing}; found {list(df.columns)}")
    for col in df.columns:
        df[col] = df[col].map(clean_text)
    return df


def collapse_export_rows(df: pd.DataFrame) -> tuple["OrderedDict[str, dict]", list[dict]]:
    """Group export rows by ID. Returns ``(by_id, inconsistencies)`` where each ``by_id`` value holds the
    scalar fields of the first row, the union of levels and the ordered list of distinct aspects, and
    ``inconsistencies`` lists IDs whose repeated rows disagree on name / wording / purpose."""
    by_id: "OrderedDict[str, dict]" = OrderedDict()
    issues: list[dict] = []
    compared = ((EXPORT_NAME, "name"), (EXPORT_WORDING_EN, "wording_en"), (EXPORT_WORDING_FR, "wording_fr"), (EXPORT_PURPOSE, "purpose"))
    for row in df.to_dict("records"):
        uid = clean_text(row.get(EXPORT_ID))
        if not uid:
            log.warning("export row without Unique ID skipped: %r", row.get(EXPORT_NAME))
            continue
        aspect = Aspect(section=clean_text(row.get(EXPORT_SECTION)), sector=clean_text(row.get(EXPORT_SECTOR)), sub_sector=clean_text(row.get(EXPORT_SUB_SECTOR)))
        if uid not in by_id:
            by_id[uid] = {
                "indikit_id": uid,
                "name": clean_text(row.get(EXPORT_NAME)),
                "wording_en": clean_text(row.get(EXPORT_WORDING_EN)),
                "wording_fr": clean_text(row.get(EXPORT_WORDING_FR)),
                "purpose": clean_text(row.get(EXPORT_PURPOSE)),
                "guidance": clean_text(row.get(EXPORT_GUIDANCE)),
                "levels": normalise_levels(row.get(EXPORT_LEVEL)),
                "aspects": [aspect],
                "rows": 1,
            }
            continue
        rec = by_id[uid]
        rec["rows"] += 1
        if aspect not in rec["aspects"]:
            rec["aspects"].append(aspect)
        rec["levels"] = normalise_levels(rec["levels"] + normalise_levels(row.get(EXPORT_LEVEL)))
        for col, key in compared:
            other = clean_text(row.get(col))
            if normalise_for_compare(rec[key]) != normalise_for_compare(other):
                issues.append({"indikit_id": uid, "field": key, "first_value": rec[key], "other_value": other})
    return by_id, issues


def read_scraped(path: Path = config.SCRAPED_JSON) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """``scraped.json`` -> ({id: record}, {id: [levels]}); records with ``ok=False`` are dropped."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = {str(uid): r for uid, r in data.get("records", {}).items() if r.get("ok", True)}
    levels = {str(uid): normalise_levels(v) for uid, v in data.get("levels", {}).items()}
    return records, levels


def _find_col(columns: Iterable[str], *needles: str) -> Optional[str]:
    for c in columns:
        if all(n.lower() in c.lower() for n in needles):
            return c
    return None


def read_site_csv(path: Path = config.SITE_CSV) -> dict[str, dict]:
    """The notebook scrape CSV -> {id: {url, name, sub_sector, wording_en, wording_fr, purpose, how_to_collect,
    disaggregate, important_comments}}. Its ``sector`` column holds the IndiKit *sub-sector*."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    cols = list(df.columns)
    c_word = _find_col(cols, "section:Indicator", "Wording")
    c_purp = _find_col(cols, "section:Indicator", "Purpose")
    c_how = _find_col(cols, "section:How to Collect")
    c_dis = _find_col(cols, "section:Disaggregate")
    c_imp = _find_col(cols, "section:Important Comments")
    out: dict[str, dict] = {}
    for row in df.to_dict("records"):
        slug = clean_text(row.get("slug")) or clean_text(row.get("url")).rsplit("/", 1)[-1]
        uid = id_from_url(slug)
        en, fr = split_indikit_wording(row.get(c_word, "") if c_word else "")
        out[uid] = {
            "unique_id": uid,
            "url": clean_text(row.get("url")),
            "name": clean_text(row.get("title")),
            "sub_sector": clean_text(row.get("sector")),
            "wording_en": clean_text(en),
            "wording_fr": clean_text(fr),
            "purpose": clean_text(row.get(c_purp)) if c_purp else "",
            "how_to_collect": clean_text(row.get(c_how)) if c_how else "",
            "disaggregate": clean_text(row.get(c_dis)) if c_dis else "",
            "important_comments": clean_text(row.get(c_imp)) if c_imp else "",
        }
    return out


def read_supplement(path: Path = config.SITE_SUPPLEMENT_JSON) -> dict[str, dict]:
    """Records fetched by ``fetch_missing.py`` (same shape as ``scraped.json`` records); {} if absent."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(uid): r for uid, r in data.get("records", {}).items() if r.get("ok", True)}


def seed_records(urls: Iterable[str] = config.PIP_SEED_URLS, sector: str = config.PIP_SEED_SECTOR, section: str = config.PIP_SEED_SECTION) -> dict[str, dict]:
    """Placeholder site records for indicators we know exist on the site but could not fetch."""
    out = {}
    for url in urls:
        uid = id_from_url(url)
        out[uid] = {"unique_id": uid, "url": url, "name": name_from_slug(url), "section": section, "sector": sector, "sub_sector": "", "seed": True}
    return out


def combine_site_sources(scraped: dict[str, dict], site_csv: dict[str, dict], supplement: dict[str, dict]) -> dict[str, dict]:
    """One cleaned site record per ID: scraped.json first, gaps filled from the site CSV; supplement records added."""
    keys = ("url", "name", "section", "sector", "sub_sector", "wording_en", "wording_fr", "purpose", "how_to_collect", "disaggregate", "important_comments")
    combined: dict[str, dict] = {}
    for uid, rec in list(scraped.items()) + list(supplement.items()):
        combined[uid] = {k: clean_text(rec.get(k)) for k in keys}
        combined[uid]["unique_id"] = uid
    for uid, rec in site_csv.items():
        tgt = combined.setdefault(uid, {k: "" for k in keys} | {"unique_id": uid})
        for k in keys:
            if not tgt.get(k) and rec.get(k):
                tgt[k] = clean_text(rec[k])
    return combined


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
@dataclass
class MergeResult:
    records: list[IndicatorMetadata]
    conflicts: list[dict] = field(default_factory=list)
    export_inconsistencies: list[dict] = field(default_factory=list)


def _pick(uid: str, fieldname: str, export_val: str, site_val: str, conflicts: list[dict]) -> tuple[str, Optional[str]]:
    """Site wins when both are non-empty and differ (conflict recorded); otherwise whichever is filled."""
    ev, sv = clean_text(export_val), clean_text(site_val)
    if ev and sv:
        if normalise_for_compare(ev) != normalise_for_compare(sv):
            conflicts.append({"indikit_id": uid, "field": fieldname, "export_value": ev, "site_value": sv, "resolved": "site"})
        return sv, "site"
    if sv:
        return sv, "site"
    if ev:
        return ev, "export"
    return "", None


def _merge_aspects(uid: str, export_aspects: list[Aspect], site: Optional[dict], conflicts: list[dict]) -> tuple[list[Aspect], Optional[str]]:
    if site is None or not (site.get("sector") or site.get("sub_sector")):
        return list(export_aspects), ("export" if export_aspects else None)
    s_aspect = Aspect(section=site.get("section", ""), sector=site.get("sector", ""), sub_sector=site.get("sub_sector", ""))
    if not export_aspects:
        return [s_aspect], "site"
    norm = lambda a: (normalise_for_compare(a.section), normalise_for_compare(a.sector), normalise_for_compare(a.sub_sector))  # noqa: E731
    if norm(s_aspect) in {norm(a) for a in export_aspects}:
        return list(export_aspects), "export"
    # same section/sector, export only lacks the sub-sector -> fill it in, no conflict
    for i, a in enumerate(export_aspects):
        if norm(a)[:2] == norm(s_aspect)[:2] and not a.sub_sector and s_aspect.sub_sector:
            merged = list(export_aspects)
            merged[i] = s_aspect
            return merged, "site"
    conflicts.append(
        {
            "indikit_id": uid,
            "field": "aspect",
            "export_value": "; ".join(f"{a.section} > {a.sector} > {a.sub_sector}" for a in export_aspects),
            "site_value": f"{s_aspect.section} > {s_aspect.sector} > {s_aspect.sub_sector}",
            "resolved": "site",
        }
    )
    return [s_aspect] + list(export_aspects), "site"


def merge_records(
    export_by_id: dict[str, dict],
    site_by_id: dict[str, dict],
    site_levels: Optional[dict[str, list[str]]] = None,
    seeds: Optional[dict[str, dict]] = None,
) -> MergeResult:
    """Apply the merge rules to the collapsed export and the combined site records."""
    site_levels = site_levels or {}
    seeds = {uid: r for uid, r in (seeds or {}).items() if uid not in export_by_id and uid not in site_by_id}
    conflicts: list[dict] = []
    records: list[IndicatorMetadata] = []
    for uid in sorted(set(export_by_id) | set(site_by_id) | set(seeds), key=_sort_key):
        e = export_by_id.get(uid) or {}
        s = site_by_id.get(uid)
        seed = seeds.get(uid)
        src: dict[str, str] = {}
        if seed is not None:
            s_eff = seed
        else:
            s_eff = s or {}
        site_tag = "seed" if seed is not None else "site"

        def pick(fieldname: str, e_key: str, s_key: str) -> str:
            val, where = _pick(uid, fieldname, e.get(e_key, ""), s_eff.get(s_key, ""), conflicts)
            if where:
                src[fieldname] = site_tag if where == "site" else where
            return val

        name = pick("name", "name", "name")
        wording_en = pick("kpi_wording_en", "wording_en", "wording_en")
        wording_fr = pick("kpi_wording_fr", "wording_fr", "wording_fr")
        purpose = pick("purpose", "purpose", "purpose")
        aspects, a_src = _merge_aspects(uid, e.get("aspects", []), s_eff if (s is not None or seed is not None) else None, conflicts)
        if a_src:
            src["aspect"] = site_tag if a_src == "site" else a_src

        levels_e = e.get("levels", [])
        levels_s = site_levels.get(uid, [])
        levels = normalise_levels(levels_e + levels_s)
        if levels:
            src["levels"] = "site" if levels_s else "export"

        how = clean_text(s_eff.get("how_to_collect"))
        topic = clean_text(s_eff.get("disaggregate"))
        comments = clean_text(s_eff.get("important_comments"))
        for fieldname, val in (("how_to_collect", how), ("topic", topic), ("important_comments", comments)):
            if val:
                src[fieldname] = site_tag
        guidance = clean_text(e.get("guidance"))
        if guidance:
            src["guidance"] = "export"
        url = clean_text(s_eff.get("url"))
        if url:
            src["url"] = site_tag
        elif guidance.startswith("http"):
            url = guidance
            src["url"] = "export"

        quantity = derive_quantity(wording_en, name)
        src["quantity"] = "derived"
        src["indikit_id"] = "export" if e else site_tag

        records.append(
            IndicatorMetadata(
                indikit_id=uid,
                name=name or name_from_slug(url) or uid,
                aspect=aspects,
                kpi_wording_en=wording_en,
                kpi_wording_fr=wording_fr or None,
                levels=levels,
                topic=topic or None,
                quantity=quantity,
                purpose=purpose or None,
                how_to_collect=how or None,
                important_comments=comments or None,
                guidance=guidance or None,
                url=url or None,
                on_site=s is not None,
                in_export=bool(e),
                field_sources=dict(sorted(src.items())),
            )
        )
    return MergeResult(records=records, conflicts=conflicts)


# --------------------------------------------------------------------------- #
# Writers / readers of the outputs
# --------------------------------------------------------------------------- #
def write_jsonl(records: Iterable[IndicatorMetadata], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def read_jsonl(path: Path = config.METADATA_JSONL) -> list[IndicatorMetadata]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m pipeline.run build-metadata")
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(IndicatorMetadata.model_validate_json(line))
    return out


def write_reference_library(records: Iterable[IndicatorMetadata], path: Path) -> int:
    """Legacy ``reference_library.csv`` (indicator_id, indicator_text, sector, level + a few extras)."""
    path = Path(path)
    rows = []
    for r in records:
        rows.append(
            {
                "indicator_id": r.indikit_id,
                "indicator_text": r.kpi_wording_en or r.name,
                "sector": r.primary_sector,
                "level": " | ".join(r.levels),
                "name": r.name,
                "indicator_text_fr": r.kpi_wording_fr or "",
                "sub_sector": r.primary_sub_sector,
                "url": r.url or "",
                "text_source": "wording" if r.kpi_wording_en else "name",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["indicator_id", "indicator_text", "sector", "level"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_conflicts(conflicts: list[dict], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["indikit_id", "field", "export_value", "site_value", "resolved"])
        writer.writeheader()
        writer.writerows(sorted(conflicts, key=lambda c: (_sort_key(c["indikit_id"]), c["field"])))
    return len(conflicts)


# --------------------------------------------------------------------------- #
# Build report
# --------------------------------------------------------------------------- #
@dataclass
class BuildStats:
    export_rows: int = 0
    export_unique_ids: int = 0
    site_records: int = 0
    supplement_records: int = 0
    seeded: int = 0
    total: int = 0
    both: int = 0
    export_only: int = 0
    site_only: int = 0
    seed_only: int = 0
    per_sector: dict = field(default_factory=dict)
    per_level: dict = field(default_factory=dict)
    missing_fields: dict = field(default_factory=dict)
    conflicts_by_field: dict = field(default_factory=dict)
    quantity_distribution: dict = field(default_factory=dict)
    export_inconsistencies: list = field(default_factory=list)
    missing_wording_ids: list = field(default_factory=list)
    extension_exceptions: list = field(default_factory=list)
    overrides_applied: list = field(default_factory=list)
    overrides_skipped: list = field(default_factory=list)
    enrichment: dict = field(default_factory=dict)


def compute_stats(records: list[IndicatorMetadata], conflicts: list[dict]) -> BuildStats:
    st = BuildStats(total=len(records))
    st.both = sum(1 for r in records if r.on_site and r.in_export)
    st.export_only = sum(1 for r in records if r.in_export and not r.on_site)
    st.site_only = sum(1 for r in records if r.on_site and not r.in_export)
    st.seed_only = sum(1 for r in records if r.needs_fetch)
    st.per_sector = dict(sorted(Counter(r.primary_sector or "(none)" for r in records).items(), key=lambda kv: (-kv[1], kv[0])))
    st.per_level = dict(sorted(Counter(" | ".join(r.levels) or "(missing)" for r in records).items(), key=lambda kv: (-kv[1], kv[0])))
    checked = ("kpi_wording_en", "kpi_wording_fr", "levels", "topic", "purpose", "how_to_collect", "important_comments", "url", "knowledge", "search_terms")
    st.missing_fields = {f: sum(1 for r in records if not getattr(r, f)) for f in checked}
    st.conflicts_by_field = dict(sorted(Counter(c["field"] for c in conflicts).items()))
    st.quantity_distribution = dict(sorted(Counter(r.quantity.value for r in records).items(), key=lambda kv: (-kv[1], kv[0])))
    st.missing_wording_ids = [r.indikit_id for r in records if not r.kpi_wording_en]
    return st


def _md_table(rows: Iterable[tuple], header: tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _rel(path: Path) -> str:
    """Path relative to the repository root when possible (keeps the report identical across machines)."""
    try:
        return str(Path(path).resolve().relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


def render_report(st: BuildStats, conflicts_path: Path, jsonl_path: Path) -> str:
    parts = [
        "# IndiKit metadata build report",
        "",
        "Generated by `python -m pipeline.run build-metadata` (deterministic; no timestamp on purpose).",
        "",
        "## Inputs",
        "",
        _md_table(
            [
                ("Export rows read", st.export_rows),
                ("Export unique IDs", st.export_unique_ids),
                ("Site records (scraped.json)", st.site_records),
                ("Supplement records (fetch_missing.py)", st.supplement_records),
                ("Seeded records (no fetch available)", st.seeded),
            ],
            ("Input", "Count"),
        ),
        "",
        "## Records",
        "",
        _md_table(
            [
                ("Total records", st.total),
                ("In export and on site", st.both),
                ("Export-only (not on site)", st.export_only),
                ("Site-only (not in export)", st.site_only),
                ("Seed-only (needs fetch)", st.seed_only),
            ],
            ("Category", "Count"),
        ),
        "",
        "### Records per sector (primary aspect)",
        "",
        _md_table(st.per_sector.items(), ("Sector", "Records")),
        "",
        "### Records per level combination",
        "",
        _md_table(st.per_level.items(), ("Levels", "Records")),
        "",
        "### Missing-field counts",
        "",
        _md_table(st.missing_fields.items(), ("Field", "Records with empty value")),
        "",
        f"IDs without English wording: {', '.join(st.missing_wording_ids) if st.missing_wording_ids else 'none'}",
        "",
        "### Quantity-type distribution (derived from wording)",
        "",
        _md_table(st.quantity_distribution.items(), ("Quantity", "Records")),
        "",
        "## Merge conflicts (site wins)",
        "",
        f"Total conflicts: {sum(st.conflicts_by_field.values())}; details in `{_rel(conflicts_path)}`.",
        "",
        _md_table(st.conflicts_by_field.items(), ("Field", "Conflicts")) if st.conflicts_by_field else "No conflicts.",
        "",
        "## Export rows that disagree within the same ID",
        "",
        (_md_table(((i["indikit_id"], i["field"]) for i in st.export_inconsistencies), ("ID", "Field")) if st.export_inconsistencies else "None: repeated export rows are identical apart from sector / sub-sector / level."),
        "",
        "## Extensions",
        "",
        f"Enrichment: {st.enrichment or 'not run (deterministic baseline only)'}",
        "",
        "Overrides applied: " + (", ".join(st.overrides_applied) if st.overrides_applied else "none"),
        "",
        "Overrides skipped (name mismatch or unknown ID): " + ("; ".join(f"{i} ({why})" for i, why in st.overrides_skipped) if st.overrides_skipped else "none"),
        "",
        "Records with fewer than 3 search terms or empty knowledge after the baseline:",
        "",
        (_md_table(st.extension_exceptions, ("ID", "Problem")) if st.extension_exceptions else "none"),
        "",
        f"Library written to `{_rel(jsonl_path)}`.",
        "",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class BuildResult:
    records: list[IndicatorMetadata]
    conflicts: list[dict]
    stats: BuildStats
    report: str


def build(
    export_csv: Path = config.EXPORT_CSV,
    scraped_json: Path = config.SCRAPED_JSON,
    site_csv: Optional[Path] = config.SITE_CSV,
    supplement_json: Optional[Path] = config.SITE_SUPPLEMENT_JSON,
    overrides_yaml: Optional[Path] = config.OVERRIDES_YAML,
    out_jsonl: Path = config.METADATA_JSONL,
    reference_csv: Optional[Path] = config.REFERENCE_LIBRARY_CSV,
    report_md: Optional[Path] = config.BUILD_REPORT_MD,
    conflicts_csv: Optional[Path] = config.MERGE_CONFLICTS_CSV,
    seed_urls: Iterable[str] = config.PIP_SEED_URLS,
    enrich: bool = False,
    enrich_model: Optional[str] = None,
) -> BuildResult:
    """Run the whole metadata build (entities -> extensions -> overrides -> outputs)."""
    from pipeline.metadata import extensions  # local import keeps build_metadata importable without yaml

    export_df = read_export(export_csv)
    export_by_id, inconsistencies = collapse_export_rows(export_df)
    scraped, site_levels = read_scraped(scraped_json)
    site_rows = read_site_csv(site_csv) if site_csv and Path(site_csv).exists() else {}
    supplement = read_supplement(supplement_json) if supplement_json else {}
    site_by_id = combine_site_sources(scraped, site_rows, supplement)
    seeds = seed_records(seed_urls)

    merged = merge_records(export_by_id, site_by_id, site_levels, seeds)
    records = merged.records

    exceptions = extensions.apply_baseline(records)
    enrichment_stats: dict = {}
    if enrich:
        enrichment_stats = extensions.enrich_records(records, model_name=enrich_model)
    overrides = extensions.load_overrides(overrides_yaml) if overrides_yaml else {}
    applied, skipped = extensions.apply_overrides(records, overrides)

    stats = compute_stats(records, merged.conflicts)
    stats.export_rows = len(export_df)
    stats.export_unique_ids = len(export_by_id)
    stats.site_records = len(scraped)
    stats.supplement_records = len(supplement)
    stats.seeded = sum(1 for r in records if r.needs_fetch)
    stats.export_inconsistencies = inconsistencies
    stats.extension_exceptions = exceptions
    stats.overrides_applied = applied
    stats.overrides_skipped = skipped
    stats.enrichment = enrichment_stats

    write_jsonl(records, out_jsonl)
    if reference_csv:
        write_reference_library(records, reference_csv)
    if conflicts_csv:
        write_conflicts(merged.conflicts, conflicts_csv)
    report = render_report(stats, conflicts_csv or Path("merge_conflicts.csv"), out_jsonl)
    if report_md:
        Path(report_md).parent.mkdir(parents=True, exist_ok=True)
        Path(report_md).write_text(report, encoding="utf-8")
    return BuildResult(records=records, conflicts=merged.conflicts, stats=stats, report=report)
