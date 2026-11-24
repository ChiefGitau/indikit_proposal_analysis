"""Loader for the pipeline's latest run outputs (``outputs/latest/`` of the proposal_investigated pipeline).

The app makes **no** LLM calls and needs none of the pipeline's optional dependencies: it only reads CSV / Markdown
files. The location comes from ``src.config`` (environment variable ``ZOA_PIPELINE_LATEST`` overrides it). When no
run exists every loader returns an empty frame and ``has_results()`` is False, so pages show the demo banner.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import LATEST_DIR, METADATA_JSONL, PROPOSALS_PREFIX

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


def parse_nearest(value) -> list[tuple[str, float]]:
    """``"5223 (0.55); 5240 (0.53)"`` (document_indicators.nearest_indikit) -> ``[("5223", 0.55), ("5240", 0.53)]``."""
    out = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part:
            continue
        ind, _, score = part.partition("(")
        try:
            out.append((ind.strip(), float(score.rstrip(")").strip())))
        except ValueError:
            out.append((ind.strip(), float("nan")))
    return out


def parse_nearest_pair(ids, sims) -> list[tuple[str, float]]:
    """novel_indicators.nearest_indikit_ids + nearest_similarities (two ``;``-separated columns) -> ``[(id, score)]``."""
    id_list = [x.strip() for x in str(ids or "").split(";") if x.strip()]
    sim_list = [x.strip() for x in str(sims or "").split(";")]
    out = []
    for i, ind in enumerate(id_list):
        try:
            out.append((ind, float(sim_list[i])))
        except (IndexError, ValueError):
            out.append((ind, float("nan")))
    return out


@st.cache_data
def indikit_names(_mtime: float = 0.0, _run: str = "") -> dict[str, str]:
    """IndiKit id -> name. From the metadata library when the pipeline repository is present, completed with the names
    that appear in the run's own CSVs (so suggestions resolve even when the app runs without the pipeline)."""
    names: dict[str, str] = {}
    meta = Path(METADATA_JSONL)
    if meta.exists():
        with meta.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("indikit_id") and rec.get("name"):
                    names[str(rec["indikit_id"])] = str(rec["name"])
    for key in ("portfolio_never_used_indikit", "portfolio_most_used_indikit", "disclosure_matrix", "document_indicators"):
        df = table(key)
        if {"indikit_id", "indikit_name"} <= set(df.columns):
            for i, n in zip(df["indikit_id"], df["indikit_name"]):
                if i and n and i not in names:
                    names[str(i)] = str(n)
    return names


def indikit_label(ind: str, score: float | None, names: dict[str, str]) -> str:
    """``Name (id) · 0.55`` for display; the id alone when the name is unknown."""
    base = f"{names[ind]} ({ind})" if ind in names else str(ind)
    return f"{base} · {score:.2f}" if score is not None and score == score else base


def nearest_names() -> dict[str, str]:
    """``indikit_names()`` keyed on the metadata file and the run, so a new run or library refreshes the cache."""
    meta = Path(METADATA_JSONL)
    return indikit_names(meta.stat().st_mtime if meta.exists() else 0.0, run_id())


@st.cache_data
def indikit_library(_mtime: float = 0.0, _run: str = "") -> pd.DataFrame:
    """The whole IndiKit library: indikit_id, indikit_name, sector, sub_sector, on_site, url. From the metadata
    library when the pipeline repository is present; otherwise reconstructed from the run's most-used + never-used
    tables (then without sub-sector)."""
    meta = Path(METADATA_JSONL)
    rows = []
    if meta.exists():
        with meta.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                aspect = (rec.get("aspect") or [{}])[0]
                rows.append({"indikit_id": str(rec.get("indikit_id", "")), "indikit_name": rec.get("name") or "", "sector": aspect.get("sector") or "(no sector)",
                             "sub_sector": aspect.get("sub_sector") or "(no sub-sector)", "on_site": bool(rec.get("on_site", True)), "url": rec.get("url") or ""})
    if not rows:
        for key in ("portfolio_most_used_indikit", "portfolio_never_used_indikit"):
            df = table(key)
            for _, r in df.iterrows():
                rows.append({"indikit_id": str(r["indikit_id"]), "indikit_name": r.get("indikit_name", ""), "sector": r.get("sector", "") or "(no sector)", "sub_sector": "(no sub-sector)",
                             "on_site": str(r.get("on_site", "True")).lower() != "false", "url": ""})
    lib = pd.DataFrame(rows, columns=["indikit_id", "indikit_name", "sector", "sub_sector", "on_site", "url"]).drop_duplicates("indikit_id")
    return lib[lib["indikit_id"] != ""].reset_index(drop=True)


def library() -> pd.DataFrame:
    meta = Path(METADATA_JSONL)
    return indikit_library(meta.stat().st_mtime if meta.exists() else 0.0, run_id())


def closest_document_indicators(doc_ind: pd.DataFrame, doc_labels: dict[str, str]) -> dict[str, list[tuple[float, str, str, str]]]:
    """Invert the per-row ``nearest_indikit`` lists: IndiKit id -> [(similarity, indicator_text, document label,
    doc_indicator_id)] sorted best first. Rows without a parsable similarity get -1."""
    out: dict[str, list[tuple[float, str, str, str]]] = {}
    if "nearest_indikit" not in doc_ind.columns:
        return out
    for _, r in doc_ind.iterrows():
        for ind, score in parse_nearest(r.get("nearest_indikit", "")):
            out.setdefault(ind, []).append((score if score == score else -1.0, r["indicator_text"], doc_labels.get(r["doc_id"], r["doc_id"]), r["doc_indicator_id"]))
    for v in out.values():
        v.sort(key=lambda t: t[0], reverse=True)
    return out
