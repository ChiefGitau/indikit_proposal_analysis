"""Persistent analysis log.

Every page records the tables + metrics it computes here, so the results can be
explored later (the *Analysis Log* page) and shared (one zip), independent of
which pages happened to be visited this session.

Durable store on disk under ``data/analysis_log/``::

    data/analysis_log/
      manifest.json                  # per-section metrics + table index + context
      <section>/<table>.csv          # one CSV per logged table

Each section holds the most recent run of that page, tagged with the filter
context that produced it.

Writes only happen with ``ZOA_LOCAL_MODE=1`` (see ``src/config.py``); the deployed app is read-only.
"""

import io
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import LOCAL_MODE

# Canonical location in the central data dir (shared with docs/analysis reports).
LOG_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "indicators"
    / "data_analysis"
    / "indikit_analysis_log"
)
MANIFEST = LOG_DIR / "manifest.json"


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text)).strip("_") or "x"


def _load() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save(manifest: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))


def log_section(section, *, metrics=None, tables=None, context=None, indexed_tables=()):
    """Record one section's analysis in a single write.

    ``metrics``  – dict of JSON-serialisable scalars.
    ``tables``   – dict of ``name -> DataFrame`` (written as CSV).
    ``context``  – dict describing what produced the data (e.g. active filters).
    ``indexed_tables`` – names in ``tables`` whose row index should be kept
                         (crosstabs, where the index is the row label).
    """
    if not LOCAL_MODE:  # read-only app: never write to the repository's files
        return
    manifest = _load()
    entry = manifest.setdefault(section, {"metrics": {}, "tables": {}, "context": {}})

    if context:
        entry["context"].update(context)
    if metrics:
        entry["metrics"] = {k: v for k, v in metrics.items()}
    if tables:
        sect_dir = LOG_DIR / _slug(section)
        sect_dir.mkdir(parents=True, exist_ok=True)
        # Drop tables no longer produced so stale CSVs don't linger.
        entry["tables"] = {}
        for name, df in tables.items():
            rel = f"{_slug(section)}/{_slug(name)}.csv"
            (LOG_DIR / rel).write_text(df.to_csv(index=name in indexed_tables))
            entry["tables"][name] = {"path": rel, "rows": int(len(df))}

    entry["context"]["updated"] = datetime.now().isoformat(timespec="seconds")
    _save(manifest)


def read_log() -> dict:
    return _load()


def read_table(rel_path: str) -> pd.DataFrame:
    return pd.read_csv(LOG_DIR / rel_path)


def clear_log() -> None:
    if LOCAL_MODE and LOG_DIR.exists():
        shutil.rmtree(LOG_DIR)


def export_zip() -> bytes:
    """Bundle the whole log (manifest + every CSV) into a single zip."""
    manifest = _load()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
        for entry in manifest.values():
            for info in entry.get("tables", {}).values():
                p = LOG_DIR / info["path"]
                if p.exists():
                    zf.writestr(info["path"], p.read_text())
    return buf.getvalue()
