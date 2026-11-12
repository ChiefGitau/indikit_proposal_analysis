"""Shared table clean-up used by the three parsers: cell normalisation, empty rows / columns, header choice,
key-value tables, and the ``Table`` factory."""
from __future__ import annotations

import re
from typing import Optional

from pipeline import config
from pipeline.preprocess.logframe import classify_table, header_row_index
from pipeline.preprocess.models import Table
from pipeline.preprocess.textnorm import cap_length, clean_block_text

_KV_LABEL_RE = re.compile(r"^[^:]{2,60}:?\s*$")


def clean_grid(grid: list[list[Optional[str]]], warnings: list[str], label: str = "") -> list[list[str]]:
    width = max((len(r) for r in grid), default=0)
    out: list[list[str]] = []
    for row in grid:
        cells = [clean_block_text(str(c)) if c is not None else "" for c in row]
        cells = [cap_length(c, config.MAX_CELL_CHARS, "cell", warnings, label) for c in cells]
        cells += [""] * (width - len(cells))
        out.append(cells)
    return out


def drop_empty(grid: list[list[str]]) -> list[list[str]]:
    rows = [r for r in grid if any(c.strip() for c in r)]
    if not rows:
        return []
    keep_cols = [j for j in range(len(rows[0])) if any(j < len(r) and r[j].strip() for r in rows)]
    return [[r[j] if j < len(r) else "" for j in keep_cols] for r in rows]


def looks_key_value(rows: list[list[str]]) -> bool:
    """Two-column tables like 'Project number: | SYR2043' have no header row."""
    if not rows or len(rows[0]) != 2 or len(rows) < 2:
        return False
    labels = [r[0] for r in rows]
    return sum(1 for l in labels if l and _KV_LABEL_RE.match(l) and len(l.split()) <= 8) >= 0.7 * len(rows)


def choose_header(rows: list[list[str]], scan: int = 8) -> tuple[list[str], list[list[str]], str]:
    """Return (header, data rows, caption_from_rows_above). A logframe-style header row may sit below title rows."""
    if not rows:
        return [], [], ""
    if looks_key_value(rows):
        return ["field", "value"], rows, ""
    idx = header_row_index(rows, max_scan=scan)
    if idx is None:
        idx = next((i for i, r in enumerate(rows[:scan]) if sum(1 for c in r if c.strip()) >= max(2, min(3, len(r)))), 0)
    caption = " ".join(" ".join(dict.fromkeys(c for c in r if c.strip())) for r in rows[:idx]).strip()
    header = list(rows[idx])
    data = rows[idx + 1 :]
    # multi-row header: a following short row without digits that only adds labels is merged into the header
    while data and _is_header_continuation(header, data[0]):
        header = [f"{h} {c}".strip() if c.strip() and c.strip() != h.strip() else h for h, c in zip(header, data[0])]
        data = data[1:]
    return header, data, caption[:300]


def _is_header_continuation(header: list[str], row: list[str]) -> bool:
    cells = [c.strip() for c in row]
    non_empty = [c for c in cells if c]
    if not non_empty or len(non_empty) > len(cells):
        return False
    if any(any(ch.isdigit() for ch in c) and len(c) > 12 for c in non_empty):
        return False
    if any(len(c) > 45 for c in non_empty):
        return False
    filled_header = sum(1 for h in header if h.strip())
    # continuation rows fill columns the header left empty, or repeat short labels
    fills_gap = any(c and not header[i].strip() for i, c in enumerate(cells) if i < len(header))
    return fills_gap and filled_header >= 2 and len(non_empty) <= max(2, len(cells) // 2 + 1)


def make_table(doc_id: str, index: int, grid: list[list[Optional[str]]], *, source: str, order: int, heading_path: list[str], caption: str = "",
               page_start: Optional[int] = None, page_end: Optional[int] = None, sheet_name: Optional[str] = None, warnings: Optional[list[str]] = None,
               header: Optional[list[str]] = None) -> Optional[Table]:
    """Clean a raw grid into a classified ``Table`` (None when the grid is empty)."""
    warnings = warnings if warnings is not None else []
    table_id = f"{doc_id}:t{index:03d}"
    rows = drop_empty(clean_grid(grid, warnings, table_id))
    if not rows:
        return None
    if len(rows) > config.MAX_TABLE_ROWS:
        warnings.append(f"table {table_id} truncated from {len(rows)} to {config.MAX_TABLE_ROWS} rows")
        rows = rows[: config.MAX_TABLE_ROWS]
    if header is not None:
        hdr, data, cap_rows = header, rows, ""
    else:
        hdr, data, cap_rows = choose_header(rows)
    caption = clean_block_text(" ".join(x for x in (caption, cap_rows) if x))[:300]
    table = Table(
        table_id=table_id, doc_id=doc_id, page_start=page_start, page_end=page_end, order=order, heading_path=list(heading_path), caption=caption,
        header=[h.strip() for h in hdr], rows=data, n_rows=len(data), n_cols=len(hdr), source=source, sheet_name=sheet_name,
    )
    return classify_table(table)


def table_block_text(table: Table) -> str:
    """Short text standing in for the table in the reading-order block stream."""
    head = " | ".join(h for h in table.header if h)[:300]
    cap = f"{table.caption}. " if table.caption else ""
    return clean_block_text(f"{cap}Table with columns: {head}") if head else clean_block_text(cap or "Table")
