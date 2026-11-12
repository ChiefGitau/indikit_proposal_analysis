"""Preprocessing report (docs/phase_2.md Task 8): ``outputs/kb/preprocess_report.md`` from the manifest and the
per-document ``kb_meta.json`` files."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from pipeline import config
from pipeline.preprocess.knowledge_base import load_meta
from pipeline.preprocess.models import ManifestRow
from pipeline.preprocess.registry import read_manifest


def _table(rows: Iterable[tuple], header: tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x).replace("|", "/") for x in r) + " |" for r in rows]
    return "\n".join(lines)


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) if Path(path).exists() else 0


def render_report(manifest: dict[str, ManifestRow], kb_dir: Path = config.KB_DIR, extra: dict | None = None) -> str:
    rows = sorted(manifest.values(), key=lambda r: r.relative_path)
    corpus = [r for r in rows if r.region != "template"]
    metas = {r.doc_id: load_meta(r.doc_id, kb_dir) for r in rows}
    by = lambda key: Counter(getattr(r, key) for r in corpus)  # noqa: E731
    failures = [r for r in rows if r.parse_status == "failed"]
    partial = [r for r in rows if r.parse_status == "partial"]
    no_text = [r for r in rows if not r.has_text_layer]
    nothing = [r for r in corpus if r.n_logframe_tables == 0 and r.n_logframe_sections == 0 and r.n_references == 0 and r.parse_status != "failed"]
    lf_table = sum(1 for r in corpus if r.n_logframe_tables > 0)
    lf_section_only = sum(1 for r in corpus if r.n_logframe_tables == 0 and r.n_logframe_sections > 0)
    refs_only = sum(1 for r in corpus if r.n_logframe_tables == 0 and r.n_logframe_sections == 0 and r.n_references > 0)
    largest = []
    merges = []
    for r in rows:
        m = metas.get(r.doc_id) or {}
        for t in m.get("largest_tables", []):
            largest.append((r.relative_path, t["table_id"].rsplit(":", 1)[-1], t["n_rows"], t["n_cols"]))
        for w in m.get("continuation_merges", []):
            merges.append((r.relative_path, w))
    largest.sort(key=lambda x: -(x[2] * x[3]))
    kb_counts = {"text": sum(r.n_kb_text for r in rows), "outline": sum(r.n_kb_outline for r in rows), "table": sum(r.n_kb_table for r in rows)}
    models = Counter(r.embedding_model for r in rows)
    total_build = sum(r.build_seconds for r in rows)
    parts = [
        "# Preprocessing report",
        "",
        f"Documents in manifest: {len(rows)} (corpus `proposals/`: {len(corpus)}; template folder for sanity checks: {len(rows) - len(corpus)}). Pipeline version `{config.PIPELINE_VERSION}`.",
        "",
        "## Corpus overview",
        "",
        _table(sorted(by("file_type").items()), ("File type", "Documents")),
        "",
        _table(sorted(by("country").items()), ("Country folder", "Documents")),
        "",
        _table(sorted(by("language").items(), key=lambda kv: kv[0] or "~"), ("Language (heuristic)", "Documents")),
        "",
        _table(sorted(by("parse_status").items()), ("Parse status", "Documents")),
        "",
        "### Failures and partial parses",
        "",
        _table(((r.relative_path, r.parse_status, r.error or r.warnings[:160]) for r in failures + partial), ("Document", "Status", "Error / warning")) if failures or partial else "None.",
        "",
        "### Documents without a text layer (scanned pages)",
        "",
        _table(((r.relative_path, r.warnings[:160]) for r in no_text), ("Document", "Warning")) if no_text else "None.",
        "",
        "## Logframe coverage (corpus)",
        "",
        _table([("Logframe table found", lf_table), ("Logframe section only (no table)", lf_section_only), ("References only (no local logframe)", refs_only), ("Nothing found (manual inspection)", len(nothing))], ("Category", "Documents")),
        "",
        "### Documents with no logframe table, no logframe section and no reference",
        "",
        _table(((r.relative_path, r.file_type, r.n_tables, r.n_blocks) for r in nothing), ("Document", "Type", "Tables", "Blocks")) if nothing else "None.",
        "",
        "## Per document",
        "",
        _table(((r.relative_path, r.file_type, r.n_pages_or_sheets, r.n_blocks, r.n_tables, r.n_logframe_tables, r.n_logframe_sections, r.n_references, r.indicators_elsewhere, r.indicators_also_elsewhere, r.language, r.parse_status, r.build_seconds) for r in rows),
               ("Document", "Type", "Pages/sheets", "Blocks", "Tables", "LF tables", "LF sections", "Refs", "Elsewhere", "Also elsewhere", "Lang", "Status", "Build s")),
        "",
        "## Largest tables (top 20)",
        "",
        _table(largest[:20], ("Document", "Table", "Rows", "Cols")) if largest else "None.",
        "",
        "## Continuation-table merges (PDF)",
        "",
        _table(merges, ("Document", "Merge")) if merges else "None.",
        "",
        "## Knowledge bases",
        "",
        _table([("Text entries", kb_counts["text"]), ("Outline entries", kb_counts["outline"]), ("Table entries (rows + keywords + summaries)", kb_counts["table"]), ("Embedding model(s)", ", ".join(f"{m} ({n})" for m, n in models.items())),
                ("Total build time (s)", round(total_build, 1)), ("Mean build time per document (s)", round(total_build / max(1, len(rows)), 1)), ("Disk size of outputs/kb (MB)", round(_dir_size(kb_dir) / 1e6, 1)),
                ("Disk size of outputs/kb_baseline (MB)", round(_dir_size(config.KB_BASELINE_DIR) / 1e6, 1))], ("Item", "Value")),
        "",
    ]
    if extra:
        parts += ["## Runs", "", _table(extra.items(), ("Run", "Details")), ""]
    return "\n".join(parts)


def write_report(path: Path = config.PREPROCESS_REPORT_MD, kb_dir: Path = config.KB_DIR) -> Path:
    manifest = read_manifest(Path(kb_dir) / "manifest.csv")
    runs_path = Path(kb_dir) / "runs.json"
    extra = json.loads(runs_path.read_text(encoding="utf-8")) if runs_path.exists() else None
    text = render_report(manifest, kb_dir, extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return Path(path)


def record_run(name: str, details: dict, kb_dir: Path = config.KB_DIR) -> None:
    runs_path = Path(kb_dir) / "runs.json"
    runs = json.loads(runs_path.read_text(encoding="utf-8")) if runs_path.exists() else {}
    runs[name] = json.dumps(details, ensure_ascii=False)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    runs_path.write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding="utf-8")
