"""docx parsing in document order (paper Section 3.3.1 layout analysis). Replaces ``simple_parse_docx``.

Walks the body element by element (paragraphs, tables, content controls) so tables keep their position and
heading context. Headings come from paragraph styles with a fallback for short bold / numbered lines; merged
cells are de-duplicated; nested tables are flattened into the parent cell.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph

from pipeline import config
from pipeline.preprocess.models import Block, ParseResult, Table
from pipeline.preprocess.tables import make_table, table_block_text
from pipeline.preprocess.textnorm import cap_length, clean_block_text, heading_number_prefix

_HEADING_STYLE_RE = re.compile(r"^(heading|titre|title|überschrift|kop|rubrik|encabezado|t[ií]tulo)\s*(\d)?", re.IGNORECASE)
_LIST_STYLE_RE = re.compile(r"list|bullet|puce|num[eé]rot", re.IGNORECASE)
_CAPTION_RE = re.compile(r"^(table|tableau|figure)\s*\d+", re.IGNORECASE)


def _style_name(p: Paragraph) -> str:
    try:
        return p.style.name if p.style is not None else ""
    except Exception:
        return ""


def heading_level_from_style(style_name: str) -> Optional[int]:
    m = _HEADING_STYLE_RE.match(style_name or "")
    if not m:
        return None
    if m.group(1).lower() == "title":
        return 1
    return int(m.group(2)) if m.group(2) else 1


def _is_bold(p: Paragraph) -> bool:
    runs = [r for r in p.runs if r.text.strip()]
    if not runs:
        return False
    return all(bool(r.bold) or (r.style is not None and "strong" in (r.style.name or "").lower()) for r in runs)


def fallback_heading_level(text: str, bold: bool, style_name: str) -> Optional[int]:
    """Short bold or numbered lines ('3.2 Logical framework') act as headings when no heading style is used."""
    t = text.strip()
    if not t or len(t) > 120 or t.endswith((".", ";", ",")) or _LIST_STYLE_RE.search(style_name or ""):
        return None
    num = heading_number_prefix(t)
    words = len(t.split())
    if num and re.match(r"^(\d{1,2}\.)*\d{1,2}\.?$", num) and words <= 14:
        return min(4, num.rstrip(".").count(".") + 1)
    if num and words <= 14 and bold:
        return 1
    if bold and words <= 10 and t[0].isupper():
        return 2
    return None


def _has_numbering(p: Paragraph) -> bool:
    pPr = p._p.pPr
    return pPr is not None and pPr.find(qn("w:numPr")) is not None


def _cell_text(cell, depth: int, warnings: list[str]) -> str:
    parts = [clean_block_text(par.text) for par in cell.paragraphs if par.text and par.text.strip()]
    if depth < 2:
        for nested in cell._tc.findall(qn("w:tbl")):
            grid = _table_grid(DocxTable(nested, cell), depth + 1, warnings)
            parts.append("; ".join(" | ".join(c for c in row if c) for row in grid if any(row)))
    return " ".join(p for p in parts if p)


def _table_grid(table: DocxTable, depth: int, warnings: list[str]) -> list[list[str]]:
    """Grid of cell texts; merged cells (same underlying tc, horizontally or vertically) are blank after the first."""
    grid: list[list[str]] = []
    seen: set[int] = set()
    try:
        rows = [row.cells for row in table.rows]
    except Exception as exc:  # pragma: no cover - malformed tables
        warnings.append(f"table could not be read cell-wise ({type(exc).__name__}); falling back to raw cells")
        rows = []
        for tr in table._tbl.tr_lst:
            rows.append([DocxTable(table._tbl, table._parent).cell(0, 0).__class__(tc, table) for tc in tr.tc_lst])
    for cells in rows:
        line: list[str] = []
        for cell in cells:
            key = id(cell._tc)
            if key in seen:
                line.append("")
            else:
                seen.add(key)
                line.append(_cell_text(cell, depth, warnings))
        grid.append(line)
    return grid


def _iter_body(element):
    """Yield (kind, element) for paragraphs and tables in document order, descending into content controls."""
    for child in element.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            yield "p", child
        elif tag == "tbl":
            yield "tbl", child
        elif tag in ("sdt", "sdtContent", "smartTag", "customXml", "ins"):
            content = child.find(qn("w:sdtContent")) if tag == "sdt" else child
            if content is not None:
                yield from _iter_body(content)


def parse_docx(path: Path, doc_id: str) -> ParseResult:
    path = Path(path)
    warnings: list[str] = []
    doc = Document(str(path))
    blocks: list[Block] = []
    tables: list[Table] = []
    heading_stack: list[tuple[int, str]] = []
    order = 0
    pending_caption = ""
    last_block_was_caption = False

    def heading_path() -> list[str]:
        return [t for _, t in heading_stack]

    for kind, el in _iter_body(doc.element.body):
        if kind == "p":
            p = Paragraph(el, doc)
            text = clean_block_text(p.text)
            if not text:
                continue
            text = cap_length(text, config.MAX_BLOCK_CHARS, "block", warnings, f"{doc_id}:b{order:05d}")
            style = _style_name(p)
            level = heading_level_from_style(style)
            bold = _is_bold(p)
            if level is None and not _has_numbering(p) and not style.lower().startswith("toc"):
                level = fallback_heading_level(text, bold, style)
            if level is not None and not _CAPTION_RE.match(text):
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                blocks.append(Block(block_id=f"{doc_id}:b{order:05d}", doc_id=doc_id, kind="heading", text=text, order=order, heading_path=heading_path(), style_info={"style_name": style, "heading_level": level, "bold": bold}))
                heading_stack.append((level, text))
                order += 1
                last_block_was_caption = False
                continue
            is_caption = style.lower().startswith("caption") or bool(_CAPTION_RE.match(text))
            kind_ = "list_item" if (_has_numbering(p) or _LIST_STYLE_RE.search(style)) else "paragraph"
            blocks.append(Block(block_id=f"{doc_id}:b{order:05d}", doc_id=doc_id, kind=kind_, text=text, order=order, heading_path=heading_path(), style_info={"style_name": style, "bold": bold, "caption": is_caption}))
            order += 1
            if is_caption:
                pending_caption = text
                last_block_was_caption = True
            else:
                last_block_was_caption = False
                pending_caption = ""
        else:
            grid = _table_grid(DocxTable(el, doc), 0, warnings)
            table = make_table(doc_id, len(tables), grid, source="docx", order=order, heading_path=heading_path(), caption=pending_caption if last_block_was_caption else "", warnings=warnings)
            pending_caption = ""
            last_block_was_caption = False
            if table is None:
                continue
            tables.append(table)
            blocks.append(Block(block_id=f"{doc_id}:b{order:05d}", doc_id=doc_id, kind="table", text=table_block_text(table), order=order, heading_path=heading_path(), style_info={"table_id": table.table_id}))
            order += 1

    n_chars = sum(len(b.text) for b in blocks if b.kind != "table") + sum(len(c) for t in tables for r in t.rows for c in r)
    status = "ok" if n_chars > 0 else "partial"
    if n_chars == 0:
        warnings.append("no text extracted from docx")
    return ParseResult(blocks=blocks, tables=tables, warnings=warnings, n_pages_or_sheets=1, has_text_layer=True, status=status)
