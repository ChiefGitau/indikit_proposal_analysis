"""Gold-set loader and validator (paper Section 5.1.1; format in ``gold/README.md``).

Checks: required columns, enum values, IDs exist in the metadata, no duplicate ``doc_indicator_id``,
``match_type == none`` if and only if ``indikit_id`` is empty, index rows point at existing files.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from pipeline import config

INDEX_COLUMNS = ("doc_id", "source_file", "language", "has_logframe", "indicators_in_other_document", "labelled_by", "verified", "split")
ROW_COLUMNS = ("doc_indicator_id", "indicator_text", "level", "baseline", "target", "unit", "location", "indikit_id", "match_type")


def _blank_to_none(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None


class GoldIndexEntry(BaseModel):
    doc_id: str
    source_file: str
    language: str
    has_logframe: bool
    indicators_in_other_document: Optional[str] = None
    labelled_by: str = ""
    verified: bool = False
    split: Literal["dev", "test"]

    @field_validator("indicators_in_other_document", mode="before")
    @classmethod
    def _blank(cls, v):
        return _blank_to_none(v)

    @field_validator("doc_id", "source_file", "language", "labelled_by", mode="before")
    @classmethod
    def _strip(cls, v):
        return str(v).strip() if v is not None else v

    @field_validator("split", mode="before")
    @classmethod
    def _split_lower(cls, v):
        return str(v).strip().lower() if v is not None else v


class GoldRow(BaseModel):
    doc_indicator_id: str
    indicator_text: str
    level: Literal["impact", "outcome", "output", "unclear"]
    baseline: Optional[str] = None
    target: Optional[str] = None
    unit: Optional[str] = None
    location: Optional[str] = None
    indikit_id: Optional[str] = None
    match_type: Literal["verbatim", "close", "none"]

    @field_validator("baseline", "target", "unit", "location", "indikit_id", mode="before")
    @classmethod
    def _blank(cls, v):
        return _blank_to_none(v)

    @field_validator("doc_indicator_id", "indicator_text", mode="before")
    @classmethod
    def _required(cls, v):
        s = _blank_to_none(v)
        if s is None:
            raise ValueError("must not be empty")
        return s

    @field_validator("level", "match_type", mode="before")
    @classmethod
    def _lower(cls, v):
        return str(v).strip().lower() if v is not None else v

    @model_validator(mode="after")
    def _none_iff_no_id(self):
        if (self.match_type == "none") != (self.indikit_id is None):
            raise ValueError("match_type must be 'none' exactly when indikit_id is empty")
        return self


class GoldValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass
class GoldDocument:
    doc_id: str
    rows: list[GoldRow]
    entry: Optional[GoldIndexEntry] = None


@dataclass
class ValidationReport:
    files_checked: list[str] = field(default_factory=list)
    rows: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip() for h in (reader.fieldnames or [])]
        rows = [{(k or "").strip(): v for k, v in r.items()} for r in reader]
    return header, rows


def _fmt_error(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'row'}: {e['msg']}" for e in exc.errors())


def load_metadata_ids(jsonl_path: Path = config.METADATA_JSONL) -> set[str]:
    """IDs of the built metadata library (empty set if it has not been built yet)."""
    from pipeline.metadata.build_metadata import read_jsonl

    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return set()
    return {r.indikit_id for r in read_jsonl(jsonl_path)}


def validate_gold_file(path: Path, metadata_ids: Optional[set[str]] = None) -> tuple[list[GoldRow], list[str]]:
    """Validate one ``<doc_id>.csv``. Returns (valid rows, error messages)."""
    path = Path(path)
    errors: list[str] = []
    rows: list[GoldRow] = []
    if not path.exists():
        return rows, [f"{path.name}: file not found"]
    header, raw_rows = _read_csv(path)
    missing = [c for c in ROW_COLUMNS if c not in header]
    if missing:
        return rows, [f"{path.name}: missing columns {missing}"]
    seen: set[str] = set()
    for n, raw in enumerate(raw_rows, start=2):
        try:
            row = GoldRow.model_validate(raw)
        except ValidationError as exc:
            errors.append(f"{path.name} line {n}: {_fmt_error(exc)}")
            continue
        if row.doc_indicator_id in seen:
            errors.append(f"{path.name} line {n}: duplicate doc_indicator_id '{row.doc_indicator_id}'")
        seen.add(row.doc_indicator_id)
        if metadata_ids is not None and row.indikit_id is not None and row.indikit_id not in metadata_ids:
            errors.append(f"{path.name} line {n}: indikit_id '{row.indikit_id}' not in the metadata library")
        rows.append(row)
    return rows, errors


def load_index(path: Path = config.GOLD_INDEX_CSV) -> tuple[list[GoldIndexEntry], list[str]]:
    path = Path(path)
    if not path.exists():
        return [], [f"{path.name}: file not found"]
    header, raw_rows = _read_csv(path)
    missing = [c for c in INDEX_COLUMNS if c not in header]
    if missing:
        return [], [f"{path.name}: missing columns {missing}"]
    entries: list[GoldIndexEntry] = []
    errors: list[str] = []
    seen: set[str] = set()
    for n, raw in enumerate(raw_rows, start=2):
        try:
            entry = GoldIndexEntry.model_validate(raw)
        except ValidationError as exc:
            errors.append(f"{path.name} line {n}: {_fmt_error(exc)}")
            continue
        if entry.doc_id in seen:
            errors.append(f"{path.name} line {n}: duplicate doc_id '{entry.doc_id}'")
        seen.add(entry.doc_id)
        entries.append(entry)
    return entries, errors


def validate_gold(gold_dir: Path = config.GOLD_DIR, metadata_ids: Optional[set[str]] = None, include_templates: bool = True) -> ValidationReport:
    """Validate ``index.csv``, every file it lists and (optionally) the ``_*.csv`` templates."""
    gold_dir = Path(gold_dir)
    report = ValidationReport()
    entries, errors = load_index(gold_dir / "index.csv")
    report.files_checked.append("index.csv")
    report.errors.extend(errors)
    files = [gold_dir / f"{e.doc_id}.csv" for e in entries]
    if include_templates:
        files += sorted(p for p in gold_dir.glob("_*.csv"))
    for path in files:
        rows, errs = validate_gold_file(path, metadata_ids)
        report.files_checked.append(path.name)
        report.rows += len(rows)
        report.errors.extend(errs)
    return report


def load_gold(gold_dir: Path = config.GOLD_DIR, metadata_ids: Optional[set[str]] = None) -> list[GoldDocument]:
    """Load all labelled documents listed in the index; raises GoldValidationError on any problem."""
    gold_dir = Path(gold_dir)
    entries, errors = load_index(gold_dir / "index.csv")
    docs: list[GoldDocument] = []
    for entry in entries:
        rows, errs = validate_gold_file(gold_dir / f"{entry.doc_id}.csv", metadata_ids)
        errors.extend(errs)
        docs.append(GoldDocument(doc_id=entry.doc_id, rows=rows, entry=entry))
    if errors:
        raise GoldValidationError(errors)
    return docs
