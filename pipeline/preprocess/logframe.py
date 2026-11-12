"""Logframe detection (docs/phase_2.md Task 3): header-role mapping (EN/FR), logframe tables and sections,
carried-down result context per row, and ``logframe_content(doc_id)`` for the Phase 3 novel-indicator pass.

Extends the keyword lists of the legacy extraction (``INDICATOR_KEYWORDS`` / ``TABLE_HEADER_PATTERNS``).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from pipeline import config
from pipeline.agent.novel_pass import INDICATOR_KEYWORDS, TABLE_HEADER_PATTERNS  # noqa: F401  (reused, extended below)
from pipeline.preprocess.models import Block, Table, TableRow

ROLE_ORDER = ("indicator", "baseline", "target", "mov", "assumption", "result")

HEADER_ROLE_PATTERNS: dict[str, list[str]] = {
    "indicator": [r"\bindicat(?:or|eur)s?\b", r"\bovi\b", r"objectively\s+verifiable", r"objectivement\s+v[ée]rifiables?", r"\biov\b"],
    "baseline": [r"\bbase-?\s?line\b", r"valeurs?\s+de\s+r[ée]f[ée]rence", r"ligne\s+de\s+base", r"situation\s+de\s+r[ée]f[ée]rence", r"valeur\s+initiale", r"\bbaseline\b", r"\br[ée]f[ée]rence\b"],
    "target": [r"\btargets?\b", r"\bcibles?\b", r"valeurs?\s+cibles?", r"\bmilestones?\b", r"\bjalons?\b", r"objectifs?\s+chiffr[ée]s?"],
    "mov": [r"means?\s+of\s+verification", r"sources?\s+of\s+verification", r"sources?\s+et\s+moyens\s+de\s+v[ée]rification", r"moyens?\s+de\s+v[ée]rification", r"\bmov\b", r"\bmdv\b", r"sources?\s+of\s+data", r"sources?\s+de\s+donn[ée]es", r"data\s+sources?", r"verification"],
    "assumption": [r"\bassumptions?\b", r"\bhypoth[èe]ses?\b", r"\brisks?\b", r"\brisques?\b"],
    "result": [r"\boutcomes?\b", r"\boutputs?\b", r"\bimpacts?\b", r"\bresults?\b", r"\br[ée]sultats?\b", r"\bobjecti(?:f|ve)s?\b", r"intervention\s+logic", r"logique\s+d'?intervention", r"results?\s+chain", r"result\s+level", r"\bgoal\b", r"\bpurpose\b", r"\bactivit(?:y|ies|[ée]s?)\b", r"cha[iî]ne\s+de\s+r[ée]sultats"],
}
_ROLE_RE = {role: re.compile("|".join(pats), re.IGNORECASE) for role, pats in HEADER_ROLE_PATTERNS.items()}

LOGFRAME_TITLE_PATTERNS = [
    r"logical\s+framework", r"log-?\s?frame", r"results?\s+framework", r"cadre\s+logique", r"cadre\s+de\s+r[ée]sultats",
    r"cadre\s+de\s+performance", r"indicator\s+table", r"performance\s+indicators?", r"indicators?\s+matrix", r"indicator\s+tracking",
    r"\bm&?e\s+plan\b", r"\bmeal\s+plan\b", r"monitoring\s+(?:and|&)\s+evaluation\s+plan", r"plan\s+de\s+suivi", r"performance\s+monitoring\s+plan",
    r"\bpmp\b", r"\biptt\b", r"key\s+indicators?", r"indicateurs?\s+cl[ée]s?", r"indicateurs?\s+de\s+(?:performance|r[ée]sultats?)", r"theory\s+of\s+change\s+indicators?",
    r"objectives?\s+and\s+indicators?", r"indicators?\s+and\s+targets?", r"selected\s+indicators?", r"indicators?\s+per\s+(?:sector|sub-?sector|outcome|output)",
    r"list\s+of\s+indicators?", r"indicateurs?\s+(?:s[ée]lectionn[ée]s|retenus)", r"liste\s+des\s+indicateurs", r"indicator\s+reference\s+sheets?",
]
LOGFRAME_TITLE_RE = re.compile("|".join(LOGFRAME_TITLE_PATTERNS), re.IGNORECASE)

# Lines that announce a result row inside the first column of a logframe
_RESULT_ROW_RE = re.compile(r"^\s*(overall\s+objective|specific\s+objective|objectif\s+(?:g[ée]n[ée]ral|sp[ée]cifique|global)|impact|outcome|output|result|r[ée]sultat|goal|purpose|so\s*\d|io\s*\d|op\s*\d|r\s*\d(?:\.\d)*)\b", re.IGNORECASE)
_INDICATOR_LINE_RE = re.compile(r"^\s*(indicat(?:or|eur)\s*[\dA-Z.]*|#|%|number\s+of|nombre\s+de|proportion|percentage|taux|[A-Z]{1,4}\d{1,2}\b)", re.IGNORECASE)


def map_header_roles(cells: list[str]) -> list[Optional[str]]:
    """Role per header cell (first matching role in ``ROLE_ORDER``), None when nothing matches."""
    roles: list[Optional[str]] = []
    for cell in cells:
        c = (cell or "").strip()
        role = None
        if c and len(c) <= 120:
            for r in ROLE_ORDER:
                if _ROLE_RE[r].search(c):
                    role = r
                    break
        roles.append(role)
    return roles


def is_logframe_header(roles: list[Optional[str]]) -> bool:
    present = {r for r in roles if r}
    return "indicator" in present and len(present) >= 2


def header_row_index(rows: list[list[str]], max_scan: int = 8, min_cells: int = 3) -> Optional[int]:
    """Index of the first row (within ``max_scan``) that has ≥ ``min_cells`` non-empty cells and maps to ≥ 2 roles."""
    for i, row in enumerate(rows[:max_scan]):
        non_empty = [c for c in row if c and c.strip()]
        if len(non_empty) < min_cells:
            continue
        roles = {r for r in map_header_roles(row) if r}
        if len(roles) >= 2:
            return i
    return None


def title_is_logframe(*texts: Optional[str]) -> bool:
    return any(LOGFRAME_TITLE_RE.search(t) for t in texts if t)


def classify_table(table: Table) -> Table:
    """Set ``column_roles`` and ``is_logframe`` on a table (header rule or title rule)."""
    roles = map_header_roles(table.header)
    table.column_roles = roles
    by_header = is_logframe_header(roles)
    by_title = title_is_logframe(table.caption, table.sheet_name, *table.heading_path)
    table.is_logframe = bool(by_header or (by_title and any(roles)))
    if by_title and not any(roles) and table.n_cols >= 2:
        # titled "logframe" but header unrecognised (e.g. header row lost): still treat as logframe content
        table.is_logframe = True
    return table


def carry_result_context(table: Table) -> list[str]:
    """Per row, the last non-empty result cell seen so far (outcome / output / objective), carried downwards.
    Without a result column, a row whose first cell announces a result (and that has no indicator text) sets the context."""
    roles = table.column_roles or map_header_roles(table.header)
    result_cols = [i for i, r in enumerate(roles) if r == "result"]
    indicator_cols = [i for i, r in enumerate(roles) if r == "indicator"]
    context = ""
    out: list[str] = []
    for row in table.rows:
        if result_cols:
            vals = [row[i].strip() for i in result_cols if i < len(row) and row[i].strip()]
            candidate = " / ".join(vals)
            # a result column that also carries indicator rows (HOPE-style sheets): keep the last real result
            if candidate and (_RESULT_ROW_RE.match(candidate) or not _INDICATOR_LINE_RE.match(candidate)):
                context = candidate
        else:
            first = row[0].strip() if row else ""
            ind_text = " ".join(row[i] for i in indicator_cols if i < len(row)).strip()
            if first and _RESULT_ROW_RE.match(first) and not (ind_text and not _RESULT_ROW_RE.match(ind_text)):
                context = first
        out.append(context)
    return out


def dedupe_headers(header: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, h in enumerate(header):
        name = (h or "").strip() or f"col{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        out.append(name)
    return out


def build_rows(table: Table) -> list[TableRow]:
    """TableRow records with ``header: value`` text and the carried-down result context."""
    headers = dedupe_headers(table.header)
    contexts = carry_result_context(table)
    rows: list[TableRow] = []
    for i, row in enumerate(table.rows):
        cells = {headers[j]: (row[j] if j < len(row) else "") for j in range(len(headers))}
        parts = [f"{h}: {v.strip()}" for h, v in cells.items() if v and v.strip()]
        rows.append(TableRow(row_id=f"{table.table_id}:r{i:04d}", table_id=table.table_id, row_index=i, cells=cells, text=" | ".join(parts), result_context=contexts[i] if i < len(contexts) else ""))
    return rows


def flag_logframe_sections(blocks: list[Block], tables: list[Table]) -> list[str]:
    """Mark blocks under a logframe-titled heading (until the next heading of the same or higher level) and tables
    whose heading path or caption is a logframe title. Returns the ids of the logframe section headings."""
    section_ids: list[str] = []
    active_level: Optional[int] = None
    for b in blocks:
        if b.kind == "heading":
            level = int(b.style_info.get("heading_level", 1) or 1)
            if active_level is not None and level <= active_level:
                active_level = None
            if active_level is None and title_is_logframe(b.text):
                active_level = level
                section_ids.append(b.block_id)
                b.is_logframe = True
                continue
        if active_level is not None:
            b.is_logframe = True
    for t in tables:
        if t.is_logframe or title_is_logframe(t.caption, t.sheet_name, *t.heading_path):
            t.is_logframe = True
    table_ids = {t.table_id for t in tables if t.is_logframe}
    for b in blocks:
        if b.kind == "table" and b.style_info.get("table_id") in table_ids:
            b.is_logframe = True
    return section_ids


def indicator_like_line(text: str) -> bool:
    return bool(_INDICATOR_LINE_RE.match(text or ""))


# --------------------------------------------------------------------------- #
# Phase 3 interface
# --------------------------------------------------------------------------- #
def logframe_content(doc_id: str, kb_dir: Path = config.KB_DIR) -> dict:
    """Logframe tables (with rows and column roles) and logframe text blocks of a document, in reading order.

    Returns ``{"doc_id", "tables": [{"table": Table, "rows": [TableRow]}], "blocks": [Block], "section_heading_ids": [...]}``.
    """
    folder = Path(kb_dir) / doc_id
    tables_path, blocks_path = folder / "tables.jsonl", folder / "blocks.jsonl"
    if not tables_path.exists() or not blocks_path.exists():
        raise FileNotFoundError(f"no knowledge base for {doc_id} under {folder}; run: python -m pipeline.run build-kb --doc <path>")
    tables = [Table.model_validate_json(line) for line in tables_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    blocks = [Block.model_validate_json(line) for line in blocks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows_path = folder / "rows.jsonl"
    rows_by_table: dict[str, list[TableRow]] = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = TableRow.model_validate_json(line)
                rows_by_table.setdefault(r.table_id, []).append(r)
    lf_tables = [{"table": t, "rows": rows_by_table.get(t.table_id) or build_rows(t)} for t in sorted(tables, key=lambda t: t.order) if t.is_logframe]
    lf_blocks = [b for b in blocks if b.is_logframe and b.kind != "table"]
    heading_ids = [b.block_id for b in lf_blocks if b.kind == "heading" and title_is_logframe(b.text)]
    return {"doc_id": doc_id, "tables": lf_tables, "blocks": lf_blocks, "section_heading_ids": heading_ids}
