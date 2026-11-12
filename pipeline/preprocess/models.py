"""Records of the preprocessing module (paper Section 3.3).

``Block`` / ``Table`` / ``TableRow`` are what the parsers return in reading order; ``OutlineNode`` is the
heading tree; ``Reference`` a pointer to another document; ``KBEntry`` one searchable unit of a knowledge
base; ``ManifestRow`` the per-document line of ``outputs/kb/manifest.csv``.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

BlockKind = Literal["heading", "paragraph", "table", "list_item"]
FileType = Literal["pdf", "docx", "xlsx"]
ParseStatus = Literal["ok", "partial", "failed"]
KBType = Literal["text", "outline", "table"]
ReferenceKind = Literal["logframe", "indicator_table", "tracker", "other"]


class Block(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str
    doc_id: str
    kind: BlockKind
    text: str
    page: Optional[int] = None            # 1-based page (PDF); None for docx / xlsx
    order: int                            # reading-order index within the document
    heading_path: list[str] = Field(default_factory=list)
    style_info: dict = Field(default_factory=dict)   # font_size, bold, heading_level, style_name, sheet ...
    is_logframe: bool = False             # inside a logframe section (set by logframe.py)


class Table(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str
    doc_id: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    order: int = 0
    heading_path: list[str] = Field(default_factory=list)
    caption: str = ""
    header: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    source: FileType
    sheet_name: Optional[str] = None
    is_logframe: bool = False
    column_roles: list[Optional[str]] = Field(default_factory=list)   # per column: indicator | baseline | target | mov | result | assumption | None
    merged_from: list[str] = Field(default_factory=list)              # table ids merged as continuation tables (PDF)
    warnings: list[str] = Field(default_factory=list)


class TableRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_id: str
    table_id: str
    row_index: int
    cells: dict[str, str]
    text: str                              # "header: value | header: value"
    result_context: str = ""               # carried-down outcome / output cell (logframe rows)


class OutlineNode(BaseModel):
    node_id: str
    title: str
    level: int
    page: Optional[int] = None
    parent_id: Optional[str] = None
    block_ids: list[str] = Field(default_factory=list)   # blocks directly under this heading (not sub-headings)
    table_ids: list[str] = Field(default_factory=list)
    is_logframe: bool = False


class Reference(BaseModel):
    referenced_document: str
    quote: str
    location: str
    kind: ReferenceKind
    confidence: float


class KBEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    doc_id: str
    kb_type: KBType
    text_for_embedding: str
    payload_ref: dict = Field(default_factory=dict)     # block_ids / table_id / row_id / node_id / chunk block ids
    display_text: str
    location: str
    is_logframe: bool = False
    summary: Optional[str] = None                        # Phase 3 ablation hook (paper: mT5 summaries)
    entry_kind: str = "chunk"                            # chunk | node | row | keyword | summary


class ParseResult(BaseModel):
    blocks: list[Block] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    n_pages_or_sheets: int = 0
    has_text_layer: bool = True
    status: ParseStatus = "ok"


class ManifestRow(BaseModel):
    doc_id: str
    relative_path: str
    region: str
    country: str
    file_type: FileType
    sha256: str
    n_pages_or_sheets: int = 0
    n_chars: int = 0
    n_tables: int = 0
    language: str = ""
    has_text_layer: bool = True
    parse_status: ParseStatus = "ok"
    error: str = ""
    parsed_at: str = ""
    pipeline_version: str = ""
    n_blocks: int = 0
    n_logframe_tables: int = 0
    n_logframe_sections: int = 0
    n_references: int = 0
    indicators_elsewhere: bool = False
    indicators_also_elsewhere: bool = False
    build_seconds: float = 0.0
    embedding_model: str = ""
    n_kb_text: int = 0
    n_kb_outline: int = 0
    n_kb_table: int = 0
    warnings: str = ""
