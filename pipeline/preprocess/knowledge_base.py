"""Multi-type knowledge base per document (paper Section 3.3.2): text, outline and table knowledge bases with
vectors, stored under ``outputs/kb/<doc_id>/``.

Layout of a KB folder::

    blocks.jsonl  tables.jsonl  rows.jsonl  outline.json  references.json
    kb_text.jsonl  kb_outline.jsonl  kb_table.jsonl
    index_text.npy/.ids.json  index_outline.*  index_table.*
    kb_meta.json

- Text KB: section-aware chunks (consecutive paragraphs under the same heading, ~250 words, never crossing a
  top-level heading), heading path prefixed to the embedded text.
- Outline KB: one entry per heading node; expands to the section's chunks and tables.
- Table KB (one-to-many): per table one entry per row (with the carried-down result context), keyword entries
  (header cells, indicator-column cells, first-column cells, caption) and one summary entry, all pointing to
  the same table.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline import config
from pipeline.preprocess.embeddings import Embedder, EmbeddingCache, VectorIndex, get_embedder
from pipeline.preprocess.logframe import build_rows, flag_logframe_sections
from pipeline.preprocess.models import Block, KBEntry, ManifestRow, OutlineNode, ParseResult, Reference, Table, TableRow
from pipeline.preprocess.outline import build_outline, read_outline, write_outline
from pipeline.preprocess.references import detect_references, elsewhere_flags, write_references
from pipeline.preprocess.registry import DocumentRef, now_iso
from pipeline.preprocess.textnorm import detect_language

log = logging.getLogger(__name__)

KB_TYPES = ("text", "outline", "table")
MAX_KEYWORDS_PER_TABLE = 400
MAX_KEYWORD_CHARS = 160


# --------------------------------------------------------------------------- #
# Parsing dispatch
# --------------------------------------------------------------------------- #
def parse_document(ref: DocumentRef, *, ocr: bool = False) -> ParseResult:
    if ref.file_type == "pdf":
        from pipeline.preprocess.parse_pdf import parse_pdf

        return parse_pdf(ref.path, ref.doc_id, ocr=ocr)
    if ref.file_type == "docx":
        from pipeline.preprocess.parse_docx import parse_docx

        return parse_docx(ref.path, ref.doc_id)
    if ref.file_type == "xlsx":
        from pipeline.preprocess.parse_xlsx import parse_xlsx

        return parse_xlsx(ref.path, ref.doc_id)
    raise ValueError(f"unsupported file type: {ref.file_type}")


# --------------------------------------------------------------------------- #
# Entry construction
# --------------------------------------------------------------------------- #
def _words(text: str) -> int:
    return len(text.split())


def _split_long(text: str, max_words: int) -> list[str]:
    sentences: list[str] = []
    for s in re.split(r"(?<=[.!?;])\s+", text):
        words = s.split()
        if len(words) > max_words:  # no sentence punctuation: cut by word count
            sentences.extend(" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words))
        else:
            sentences.append(s)
    out, cur, n = [], [], 0
    for s in sentences:
        w = _words(s)
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out or [text]


def _location(blocks: list[Block]) -> str:
    pages = sorted({b.page for b in blocks if b.page})
    if pages:
        return f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"
    sheet = blocks[0].style_info.get("sheet") if blocks else None
    if sheet:
        return f"sheet {sheet}"
    return " > ".join(blocks[0].heading_path[-2:]) if blocks and blocks[0].heading_path else "document"


def build_text_entries(doc_id: str, blocks: list[Block], target: int = config.CHUNK_TARGET_WORDS, max_words: int = config.CHUNK_MAX_WORDS) -> tuple[list[KBEntry], dict[str, str]]:
    """Section-aware chunks. Returns (entries, block_id -> entry_id)."""
    entries: list[KBEntry] = []
    block_to_entry: dict[str, str] = {}
    group: list[Block] = []
    group_words = 0

    def flush() -> None:
        nonlocal group, group_words
        if not group:
            return
        text = "\n".join(b.text for b in group)
        pieces = _split_long(text, max_words) if _words(text) > max_words else [text]
        for piece in pieces:
            eid = f"{doc_id}:text:{len(entries):05d}"
            path = " > ".join(group[0].heading_path)
            entries.append(
                KBEntry(entry_id=eid, doc_id=doc_id, kb_type="text", entry_kind="chunk", text_for_embedding=(f"{path}\n{piece}" if path else piece), display_text=piece,
                        location=_location(group) + (f" | {path}" if path else ""), is_logframe=any(b.is_logframe for b in group), payload_ref={"block_ids": [b.block_id for b in group]})
            )
            for b in group:
                block_to_entry.setdefault(b.block_id, eid)
        group, group_words = [], 0

    for b in blocks:
        if b.kind == "heading":
            # a heading closes the current chunk only when it is top-level or the chunk is already half full
            level = int(b.style_info.get("heading_level", 1) or 1)
            if group and (level == 1 or group_words >= target // 2):
                flush()
            continue
        if b.kind == "table":
            continue
        w = _words(b.text)
        same_top = (not group) or (group[0].heading_path[:1] == b.heading_path[:1])
        if group and (not same_top or group_words + w > max_words or group_words >= target):
            flush()
        group.append(b)
        group_words += w
    flush()
    return entries, block_to_entry


def build_outline_entries(doc_id: str, nodes: list[OutlineNode], block_to_entry: dict[str, str]) -> list[KBEntry]:
    by_id = {n.node_id: n for n in nodes}
    entries: list[KBEntry] = []
    for n in nodes:
        if n.level == 0:
            continue
        parents: list[str] = []
        cur = by_id.get(n.parent_id) if n.parent_id else None
        while cur is not None and cur.level > 0:
            parents.append(cur.title)
            cur = by_id.get(cur.parent_id) if cur.parent_id else None
        parents.reverse()
        chunk_ids = sorted({block_to_entry[b] for b in n.block_ids if b in block_to_entry})
        text = n.title + (f" (in: {' > '.join(parents)})" if parents else "")
        entries.append(KBEntry(entry_id=f"{doc_id}:outline:{n.node_id.rsplit(':', 1)[-1]}", doc_id=doc_id, kb_type="outline", entry_kind="node", text_for_embedding=text, display_text=" > ".join(parents + [n.title]),
                               location=f"p.{n.page}" if n.page else "heading", is_logframe=n.is_logframe, payload_ref={"node_id": n.node_id, "chunk_entry_ids": chunk_ids, "table_ids": list(n.table_ids)}))
    return entries


def table_location(t: Table) -> str:
    if t.page_start:
        return f"p.{t.page_start}" + (f"-{t.page_end}" if t.page_end and t.page_end != t.page_start else "")
    if t.sheet_name:
        return f"sheet {t.sheet_name}"
    return " > ".join(t.heading_path[-2:]) or "document"


def table_title(t: Table) -> str:
    return t.caption or (" > ".join(t.heading_path[-2:]) if t.heading_path else "") or (f"sheet {t.sheet_name}" if t.sheet_name else "table")


def row_display_text(t: Table, row: TableRow) -> str:
    """Stand-alone text for the LLM: table title, header, carried result context and the row."""
    header = " | ".join(h for h in t.header if h)
    lines = [f"Table: {table_title(t)} [{table_location(t)}]", f"Columns: {header}"]
    if row.result_context:
        lines.append(f"Result context: {row.result_context}")
    lines.append(f"Row: {row.text}")
    return "\n".join(lines)


def build_table_entries(doc_id: str, tables: list[Table], rows_by_table: dict[str, list[TableRow]]) -> list[KBEntry]:
    entries: list[KBEntry] = []
    for t in tables:
        tid = t.table_id.rsplit(":", 1)[-1]
        loc = table_location(t)
        rows = rows_by_table.get(t.table_id, [])
        for r in rows:
            if not r.text.strip():
                continue
            emb = (f"{r.result_context} | " if r.result_context else "") + r.text
            entries.append(KBEntry(entry_id=f"{doc_id}:table:{tid}:r{r.row_index:04d}", doc_id=doc_id, kb_type="table", entry_kind="row", text_for_embedding=emb[:3000], display_text=row_display_text(t, r), location=loc,
                                   is_logframe=t.is_logframe, payload_ref={"table_id": t.table_id, "row_id": r.row_id}))
        # keyword entries
        keywords: dict[str, Optional[str]] = {}

        def add_kw(text: str, row_id: Optional[str]) -> None:
            k = re.sub(r"\s+", " ", text or "").strip()
            if 3 <= len(k) <= MAX_KEYWORD_CHARS and k.lower() not in {x.lower() for x in keywords} and len(keywords) < MAX_KEYWORDS_PER_TABLE:
                keywords[k] = row_id

        if t.caption:
            add_kw(t.caption, None)
        for h in t.header:
            add_kw(h, None)
        ind_cols = [i for i, role in enumerate(t.column_roles) if role == "indicator"]
        for r in rows:
            cells = list(r.cells.values())
            for i in ind_cols:
                if i < len(cells):
                    add_kw(cells[i], r.row_id)
            if cells and 0 not in ind_cols:
                add_kw(cells[0], r.row_id)
        for j, (k, row_id) in enumerate(keywords.items()):
            entries.append(KBEntry(entry_id=f"{doc_id}:table:{tid}:k{j:04d}", doc_id=doc_id, kb_type="table", entry_kind="keyword", text_for_embedding=k, display_text=k, location=loc, is_logframe=t.is_logframe,
                                   payload_ref={"table_id": t.table_id, **({"row_id": row_id} if row_id else {})}))
        header = " | ".join(h for h in t.header if h)
        summary_text = " | ".join(x for x in (t.caption, " > ".join(t.heading_path), header) if x)
        preview = "\n".join(r.text for r in rows[:12])
        entries.append(KBEntry(entry_id=f"{doc_id}:table:{tid}:s", doc_id=doc_id, kb_type="table", entry_kind="summary", text_for_embedding=summary_text[:3000] or f"table {tid}",
                               display_text=f"Table: {table_title(t)} [{loc}]\nColumns: {header}\n{t.n_rows} rows" + (f"\nFirst rows:\n{preview}" if preview else ""), location=loc, is_logframe=t.is_logframe,
                               payload_ref={"table_id": t.table_id}))
    return entries


# --------------------------------------------------------------------------- #
# KnowledgeBase object
# --------------------------------------------------------------------------- #
class KnowledgeBase:
    def __init__(self, doc_id: str, folder: Path, meta: dict, entries: dict[str, list[KBEntry]], indices: dict[str, VectorIndex],
                 tables: list[Table], rows: list[TableRow], blocks: list[Block], outline: list[OutlineNode], references: list[Reference]):
        self.doc_id, self.folder, self.meta = doc_id, Path(folder), meta
        self.entries = entries
        self.indices = indices
        self.entries_by_id: dict[str, KBEntry] = {e.entry_id: e for es in entries.values() for e in es}
        self.tables_by_id = {t.table_id: t for t in tables}
        self.rows_by_table: dict[str, list[TableRow]] = defaultdict(list)
        for r in rows:
            self.rows_by_table[r.table_id].append(r)
        self.blocks = blocks
        self.outline = outline
        self.references = references
        self._row_entry: dict[str, str] = {e.payload_ref["row_id"]: e.entry_id for e in entries.get("table", []) if e.entry_kind == "row"}
        self._table_rows_entries: dict[str, list[str]] = defaultdict(list)
        for e in entries.get("table", []):
            if e.entry_kind == "row":
                self._table_rows_entries[e.payload_ref["table_id"]].append(e.entry_id)

    @property
    def embedding_model(self) -> str:
        return self.meta.get("embedding_model", "")

    def search(self, kb_type: str, query_vec: np.ndarray, top_n: int) -> list[tuple[KBEntry, float]]:
        index = self.indices.get(kb_type)
        if index is None or not len(index):
            return []
        return [(self.entries_by_id[eid], score) for eid, score in index.search(query_vec, top_n)]

    def table_summary_entry(self, table_id: str) -> Optional[KBEntry]:
        return self.entries_by_id.get(f"{self.doc_id}:table:{table_id.rsplit(':', 1)[-1]}:s")

    def expand(self, entry: KBEntry, small_table_rows: int = config.SMALL_TABLE_ROWS) -> list[KBEntry]:
        """Outline hit -> its chunks and tables; table keyword / summary hit -> the matching row, or the whole
        table when small, else the summary; chunks and rows -> themselves."""
        if entry.kb_type == "text" or entry.entry_kind == "row":
            return [entry]
        if entry.kb_type == "outline":
            out = [self.entries_by_id[c] for c in entry.payload_ref.get("chunk_entry_ids", []) if c in self.entries_by_id]
            for tid in entry.payload_ref.get("table_ids", []):
                out.extend(self._expand_table(tid, small_table_rows))
            return out or [entry]
        tid = entry.payload_ref.get("table_id", "")
        row_id = entry.payload_ref.get("row_id")
        if row_id and row_id in self._row_entry:
            return [self.entries_by_id[self._row_entry[row_id]]]
        return self._expand_table(tid, small_table_rows) or [entry]

    def _expand_table(self, table_id: str, small_table_rows: int) -> list[KBEntry]:
        rows = self._table_rows_entries.get(table_id, [])
        if 0 < len(rows) <= small_table_rows:
            return [self.entries_by_id[r] for r in rows]
        summary = self.table_summary_entry(table_id)
        return [summary] if summary else []


# --------------------------------------------------------------------------- #
# Build / load
# --------------------------------------------------------------------------- #
def _write_jsonl(items, path: Path) -> None:
    path.write_text("".join(json.dumps(i.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n" for i in items), encoding="utf-8")


def _read_jsonl(path: Path, model):
    if not path.exists():
        return []
    return [model.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_kb(ref: DocumentRef, embedder: Embedder, cache: Optional[EmbeddingCache] = None, *, kb_dir: Path = config.KB_DIR, ocr: bool = False,
             pipeline_version: str = config.PIPELINE_VERSION) -> ManifestRow:
    """Parse one document, derive outline / logframe / references, build the three KBs and write the folder.
    Never raises on a parse failure: the manifest row carries ``parse_status='failed'`` and the error."""
    t0 = time.time()
    folder = Path(kb_dir) / ref.doc_id
    row = ManifestRow(doc_id=ref.doc_id, relative_path=ref.relative_path, region=ref.region, country=ref.country, file_type=ref.file_type, sha256=ref.sha256,
                      parsed_at=now_iso(), pipeline_version=pipeline_version, embedding_model=embedder.name)
    try:
        parsed = parse_document(ref, ocr=ocr)
    except Exception as exc:
        row.parse_status = "failed"
        row.error = f"{type(exc).__name__}: {str(exc)[:300]}"
        row.build_seconds = round(time.time() - t0, 2)
        log.warning("parse failed for %s: %s", ref.relative_path, row.error)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "kb_meta.json").write_text(json.dumps({"doc_id": ref.doc_id, "status": "failed", "error": row.error, "pipeline_version": pipeline_version}, indent=1), encoding="utf-8")
        return row

    blocks, tables, warnings = parsed.blocks, parsed.tables, list(parsed.warnings)
    section_ids = flag_logframe_sections(blocks, tables)
    outline = build_outline(blocks, tables, ref.doc_id)
    rows: list[TableRow] = []
    for t in tables:
        rows.extend(build_rows(t))
    rows_by_table: dict[str, list[TableRow]] = defaultdict(list)
    for r in rows:
        rows_by_table[r.table_id].append(r)
    references = detect_references(blocks, tables)

    text_entries, block_to_entry = build_text_entries(ref.doc_id, blocks)
    outline_entries = build_outline_entries(ref.doc_id, outline, block_to_entry)
    table_entries = build_table_entries(ref.doc_id, tables, rows_by_table)
    entries = {"text": text_entries, "outline": outline_entries, "table": table_entries}

    folder.mkdir(parents=True, exist_ok=True)
    _write_jsonl(blocks, folder / "blocks.jsonl")
    _write_jsonl(tables, folder / "tables.jsonl")
    _write_jsonl(rows, folder / "rows.jsonl")
    write_outline(outline, folder / "outline.json")
    write_references(references, folder / "references.json")
    indices: dict[str, VectorIndex] = {}
    for kb_type in KB_TYPES:
        es = entries[kb_type]
        _write_jsonl(es, folder / f"kb_{kb_type}.jsonl")
        texts = [e.text_for_embedding for e in es]
        vectors = cache.embed(embedder, texts) if cache is not None else embedder.embed(texts)
        idx = VectorIndex([e.entry_id for e in es], vectors)
        idx.save(folder / f"index_{kb_type}")
        indices[kb_type] = idx

    all_text = " ".join(b.text for b in blocks if b.kind != "table") + " " + " ".join(c for t in tables for r in t.rows for c in r)
    n_lf_tables = sum(1 for t in tables if t.is_logframe)
    elsewhere, also = elsewhere_flags(references, n_lf_tables, len(section_ids))
    row.n_pages_or_sheets = parsed.n_pages_or_sheets
    row.n_chars = len(all_text)
    row.n_tables = len(tables)
    row.n_blocks = len(blocks)
    row.language = detect_language(all_text)
    row.has_text_layer = parsed.has_text_layer
    row.parse_status = parsed.status
    row.n_logframe_tables = n_lf_tables
    row.n_logframe_sections = len(section_ids)
    row.n_references = len(references)
    row.indicators_elsewhere = elsewhere
    row.indicators_also_elsewhere = also
    row.n_kb_text, row.n_kb_outline, row.n_kb_table = len(text_entries), len(outline_entries), len(table_entries)
    row.warnings = " || ".join(warnings)[:2000]
    row.build_seconds = round(time.time() - t0, 2)
    meta = {
        "doc_id": ref.doc_id, "relative_path": ref.relative_path, "sha256": ref.sha256, "pipeline_version": pipeline_version, "status": parsed.status,
        "embedding_model": embedder.name, "dim": embedder.dim, "counts": {"blocks": len(blocks), "tables": len(tables), "rows": len(rows), "kb_text": len(text_entries), "kb_outline": len(outline_entries), "kb_table": len(table_entries)},
        "n_pages_or_sheets": parsed.n_pages_or_sheets, "has_text_layer": parsed.has_text_layer, "language": row.language, "n_logframe_tables": n_lf_tables, "logframe_section_ids": section_ids,
        "n_references": len(references), "indicators_elsewhere": elsewhere, "indicators_also_elsewhere": also, "warnings": warnings, "build_seconds": row.build_seconds, "built_at": row.parsed_at,
        "continuation_merges": [w for w in warnings if w.startswith("continuation table merged")], "largest_tables": sorted(({"table_id": t.table_id, "n_rows": t.n_rows, "n_cols": t.n_cols} for t in tables), key=lambda d: -d["n_rows"] * d["n_cols"])[:5],
    }
    (folder / "kb_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    return row


def kb_exists(doc_id: str, kb_dir: Path = config.KB_DIR) -> bool:
    return (Path(kb_dir) / doc_id / "kb_meta.json").exists() and (Path(kb_dir) / doc_id / "index_text.ids.json").exists()


def load_kb(doc_id: str, kb_dir: Path = config.KB_DIR) -> KnowledgeBase:
    """Load a knowledge base from disk (Phase 3 interface)."""
    folder = Path(kb_dir) / doc_id
    meta_path = folder / "kb_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no knowledge base for {doc_id} under {folder}; run: python -m pipeline.run build-kb --doc <path>")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") == "failed":
        raise RuntimeError(f"knowledge base for {doc_id} failed to build: {meta.get('error')}")
    entries = {k: _read_jsonl(folder / f"kb_{k}.jsonl", KBEntry) for k in KB_TYPES}
    indices = {k: VectorIndex.load(folder / f"index_{k}") for k in KB_TYPES if (folder / f"index_{k}.ids.json").exists()}
    references = [Reference.model_validate(r) for r in json.loads((folder / "references.json").read_text(encoding="utf-8"))] if (folder / "references.json").exists() else []
    outline = read_outline(folder / "outline.json") if (folder / "outline.json").exists() else []
    return KnowledgeBase(doc_id, folder, meta, entries, indices, _read_jsonl(folder / "tables.jsonl", Table), _read_jsonl(folder / "rows.jsonl", TableRow), _read_jsonl(folder / "blocks.jsonl", Block), outline, references)


def load_meta(doc_id: str, kb_dir: Path = config.KB_DIR) -> dict:
    path = Path(kb_dir) / doc_id / "kb_meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def default_embedder(spec: Optional[str] = None) -> Embedder:
    return get_embedder(spec or config.KB_EMBEDDING_MODEL)
