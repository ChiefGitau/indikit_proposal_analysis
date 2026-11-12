"""Loader for the pipeline's latest run outputs (``outputs/latest/`` of the proposal_investigated pipeline).

The app makes **no** LLM calls and needs none of the pipeline's optional dependencies: it only reads CSV / Markdown
files. The location comes from ``src.config`` (environment variable ``ZOA_PIPELINE_LATEST`` overrides it). When no
run exists every loader returns an empty frame and ``has_results()`` is False, so pages show the demo banner.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import LATEST_DIR, PROPOSALS_PREFIX

FILES = {
    "document_summary": "document_summary.csv",
    "document_indicators": "document_indicators.csv",
    "disclosure_matrix": "disclosure_matrix.csv",
    "novel_indicators": "novel_indicators.csv",
    "portfolio_by_country": "portfolio_by_country.csv",
    "portfolio_by_sector": "portfolio_by_sector.csv",
    "portfolio_most_used_indikit": "portfolio_most_used_indikit.csv",
    "portfolio_never_used_indikit": "portfolio_never_used_indikit.csv",
    "portfolio_novel_clusters": "portfolio_novel_clusters.csv",
    "review": "review.xlsx",
    "run_report": "run_report.md",
    "portfolio_summary": "portfolio_summary.md",
}

# folder names under proposals/ -> names Plotly's country-names locationmode understands
COUNTRY_NAMES = {
    "drc": "Democratic Republic of the Congo",
    "burkina": "Burkina Faso",
    "south_sudan": "South Sudan",
    "sudan": "Sudan",
    "ethiopia": "Ethiopia",
    "uganda": "Uganda",
    "burundi": "Burundi",
    "syria": "Syria",
    "yemen": "Yemen",
    "template": "(template)",
}


def latest_dir() -> Path:
    return Path(LATEST_DIR)


def has_results() -> bool:
    return (latest_dir() / FILES["document_summary"]).exists()


def country_name(folder: str) -> str:
    key = str(folder or "").strip().lower()
    return COUNTRY_NAMES.get(key, str(folder).replace("_", " ").title())


def _path(key: str) -> Path:
    return latest_dir() / FILES[key]


@st.cache_data
def load_csv(key: str, _mtime: float = 0.0) -> pd.DataFrame:
    p = _path(key)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    for col in ("country", "region"):
        if col in df.columns:
            df[col + "_name"] = df[col].map(country_name)
    return df


def _mtime(key: str) -> float:
    p = _path(key)
    return p.stat().st_mtime if p.exists() else 0.0


def table(key: str) -> pd.DataFrame:
    """Cached frame keyed on the file's modification time, so a new run is picked up without restarting."""
    return load_csv(key, _mtime(key))


def numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def document_label(row) -> str:
    rel = str(row.get("relative_path", "") or "")
    return rel.split(PROPOSALS_PREFIX, 1)[-1] if rel else str(row.get("doc_id", ""))


def read_text(key: str) -> str:
    p = _path(key)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def file_bytes(key: str) -> bytes:
    p = _path(key)
    return p.read_bytes() if p.exists() else b""


def run_id() -> str:
    p = latest_dir() / "LATEST_RUN.txt"
    return p.read_text(encoding="utf-8").splitlines()[0].strip() if p.exists() else ""
