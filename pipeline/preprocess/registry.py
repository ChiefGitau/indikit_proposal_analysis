"""Document registry (docs/phase_2.md Task 1): discovery under the document roots, stable ``doc_id``s and the
manifest ``outputs/kb/manifest.csv``.

``doc_id`` = slug of the path relative to its root + first 8 hex characters of the file's SHA-256, so identical
file names in different country folders stay distinct and a changed file gets a new id.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from pipeline import config
from pipeline.preprocess.models import ManifestRow
from pipeline.preprocess.textnorm import sha256_file, slugify

TEMP_PREFIXES = ("~$", ".~", "._")


@dataclass(frozen=True)
class DocumentRef:
    doc_id: str
    path: Path
    relative_path: str      # "<root label>/<path inside root>"
    region: str
    country: str
    file_type: str
    sha256: str


def _region_country(rel_parts: tuple[str, ...]) -> tuple[str, str]:
    """proposals: ``africa/burundi/large/x.docx`` -> (africa, burundi); template: (template, template)."""
    if len(rel_parts) >= 3:
        return rel_parts[0], rel_parts[1]
    if len(rel_parts) == 2:
        return rel_parts[0], rel_parts[0]
    return "", ""


def make_doc_id(relative_path: str, sha256: str) -> str:
    stem = Path(relative_path).with_suffix("")
    return f"{slugify(str(stem))}-{sha256[:8]}"


def discover_documents(roots: Iterable[tuple[str, Path]] = config.DOCUMENT_ROOTS, extensions: Iterable[str] = config.DOCUMENT_EXTENSIONS) -> list[DocumentRef]:
    """Recursively list documents under every root (sorted, temp files skipped)."""
    exts = {e.lower() for e in extensions}
    refs: list[DocumentRef] = []
    for label, root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.suffix.lower() not in exts or path.name.startswith(TEMP_PREFIXES):
                continue
            rel_inside = path.relative_to(root)
            relative_path = f"{label}/{rel_inside.as_posix()}"
            region, country = _region_country(rel_inside.parts) if label == "proposals" else (label, label)
            sha = sha256_file(path)
            refs.append(
                DocumentRef(
                    doc_id=make_doc_id(relative_path, sha),
                    path=path,
                    relative_path=relative_path,
                    region=region,
                    country=country,
                    file_type=path.suffix.lower().lstrip("."),
                    sha256=sha,
                )
            )
    return refs


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
MANIFEST_COLUMNS = list(ManifestRow.model_fields)


def read_manifest(path: Path = config.MANIFEST_CSV) -> dict[str, ManifestRow]:
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, ManifestRow] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            raw = {k: v for k, v in raw.items() if k in ManifestRow.model_fields and v not in (None, "")}
            row = ManifestRow.model_validate(raw)
            out[row.doc_id] = row
    return out


def write_manifest(rows: Iterable[ManifestRow], path: Path = config.MANIFEST_CSV) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r.relative_path):
            writer.writerow(row.model_dump(mode="json"))
    return path


def upsert_manifest(row: ManifestRow, path: Path = config.MANIFEST_CSV) -> None:
    rows = read_manifest(path)
    rows[row.doc_id] = row
    write_manifest(rows.values(), path)


def list_documents(path: Path = config.MANIFEST_CSV) -> list[ManifestRow]:
    """Manifest rows of every processed document (Phase 3 interface)."""
    return sorted(read_manifest(path).values(), key=lambda r: r.relative_path)


def get_document(doc_id: str, path: Path = config.MANIFEST_CSV) -> ManifestRow:
    rows = read_manifest(path)
    if doc_id not in rows:
        raise KeyError(f"{doc_id} not in {path}; run: python -m pipeline.run build-kb --all")
    return rows[doc_id]


def needs_processing(ref: DocumentRef, manifest: dict[str, ManifestRow], force: bool = False, pipeline_version: str = config.PIPELINE_VERSION) -> bool:
    """Skip documents whose SHA-256 and pipeline version are unchanged and that were built successfully."""
    if force:
        return True
    row = manifest.get(ref.doc_id)
    if row is None:
        return True
    return not (row.sha256 == ref.sha256 and row.pipeline_version == pipeline_version and row.parse_status != "failed")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def find_ref(doc_id_or_path: str, refs: Optional[list[DocumentRef]] = None) -> DocumentRef:
    """Resolve a doc_id, a relative path or a filesystem path to a DocumentRef."""
    refs = refs if refs is not None else discover_documents()
    for r in refs:
        if r.doc_id == doc_id_or_path or r.relative_path == doc_id_or_path:
            return r
    p = Path(doc_id_or_path)
    if p.exists():
        rp = p.resolve()
        for r in refs:
            if r.path.resolve() == rp:
                return r
        # a file outside the configured roots: register it ad hoc
        sha = sha256_file(rp)
        rel = f"adhoc/{rp.name}"
        return DocumentRef(doc_id=make_doc_id(rel, sha), path=rp, relative_path=rel, region="adhoc", country="adhoc", file_type=rp.suffix.lower().lstrip("."), sha256=sha)
    raise KeyError(f"no document matches {doc_id_or_path!r}")
