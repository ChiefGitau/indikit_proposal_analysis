"""Proposal-level loader.

Reads the pipeline's ``outputs/latest/document_summary.csv`` (path from ``src.config``) and exposes the two frame
shapes the Proposal Overview page uses. When no run exists it falls back to the hand-curated
``data/proposals_dummy.json`` and flags ``is_demo = True`` so the page shows a visible "demo data" banner.

``load_proposals()``    -> one row per proposal / document (metadata for the overview; keeps the dummy columns and
                           adds the pipeline's: doc_id, country_name, total_indicators, verbatim, close, none,
                           pct_matched, indicators_elsewhere, referenced_documents, note, language, parse_status).
``proposal_sites()``    -> one row per project site (lat / lon for the map markers; empty for pipeline data).
``covered_countries()`` -> country -> number of proposals (choropleth).
``is_demo()``           -> True when the dummy file is being shown.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.results import country_name, document_label, has_results, numeric, table

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "proposals_dummy.json"

# Fields shown as the overview table / cards, in display order.
OVERVIEW_COLS = [
    "id",
    "title",
    "donor",
    "sector",
    "location",
    "budget",
    "timeframe",
    "partners",
    "beneficiaries",
    "summary",
]
RESULT_COLS = ["doc_id", "country_name", "language", "parse_status", "total_indicators", "verbatim", "close", "none", "pct_matched", "indicators_elsewhere", "referenced_documents", "note", "pairs_asked", "llm_calls"]


def is_demo() -> bool:
    return not has_results()


@st.cache_data
def _raw(path=None) -> list:
    p = Path(path) if path else DATA_PATH
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _dummy_proposals(path=None) -> pd.DataFrame:
    rows = []
    for p in _raw(path):
        sites = p.get("sites", [])
        location = "; ".join(s["name"] for s in sites) if sites else "—"
        rows.append({"id": p["id"], "title": p["title"], "donor": p["donor"], "sector": p["sector"], "summary": p["summary"], "partners": p["partners"], "timeframe": p["timeframe"], "budget": p["budget"],
                     "beneficiaries": p["beneficiaries"], "countries": p.get("countries", []), "n_sites": len(sites), "location": location, "source_file": p.get("source_file", "")})
    df = pd.DataFrame(rows)
    for c in RESULT_COLS:
        df[c] = None
    df["doc_id"] = df["id"]
    df["country_name"] = df["countries"].map(lambda cs: cs[0] if cs else "")
    return df


def _pipeline_proposals() -> pd.DataFrame:
    s = table("document_summary")
    s = s[s["region"] != "template"] if "region" in s.columns else s
    s = numeric(s, ["total_indicators", "verbatim", "close", "none", "pct_matched", "pairs_asked", "llm_calls", "cost_usd"])
    df = pd.DataFrame({
        "id": s["doc_id"],
        "title": s.apply(document_label, axis=1),
        "donor": "—",
        "sector": "—",
        "summary": s["note"].where(s["note"] != "", "—"),
        "partners": "—",
        "timeframe": "—",
        "budget": "—",
        "beneficiaries": "—",
        "countries": s["country_name"].map(lambda c: [c] if c else []),
        "n_sites": 0,
        "location": s["country_name"],
        "source_file": s["relative_path"],
    })
    for c in RESULT_COLS:
        df[c] = s[c].values if c in s.columns else None
    df["indicators_elsewhere"] = s["indicators_elsewhere"].astype(str).str.lower().eq("true").values
    return df.reset_index(drop=True)


def load_proposals(path=None) -> pd.DataFrame:
    """One row per proposal with a human-readable ``location`` summary (pipeline results, else dummy data)."""
    return _dummy_proposals(path) if (path or is_demo()) else _pipeline_proposals()


def proposal_sites(path=None) -> pd.DataFrame:
    """One row per project site (only the dummy file has coordinates)."""
    cols = ["id", "title", "donor", "sector", "budget", "timeframe", "partners", "beneficiaries", "site", "lat", "lon"]
    if not (path or is_demo()):
        return pd.DataFrame(columns=cols)
    rows = []
    for p in _raw(path):
        for s in p.get("sites", []):
            rows.append({"id": p["id"], "title": p["title"], "donor": p["donor"], "sector": p["sector"], "budget": p["budget"], "timeframe": p["timeframe"], "partners": p["partners"], "beneficiaries": p["beneficiaries"],
                         "site": s["name"], "lat": s["lat"], "lon": s["lon"]})
    return pd.DataFrame(rows, columns=cols)


def covered_countries(path=None) -> pd.DataFrame:
    """Country -> number of proposals active there (for the choropleth)."""
    props = load_proposals(path)
    counts = {}
    for cs in props["countries"]:
        for c in cs or []:
            counts[c] = counts.get(c, 0) + 1
    return pd.DataFrame({"country": list(counts), "proposals": list(counts.values())}).sort_values("proposals", ascending=False).reset_index(drop=True)
