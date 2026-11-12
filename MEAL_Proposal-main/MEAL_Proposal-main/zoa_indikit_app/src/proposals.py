"""Proposal-level loader (DUMMY data).

Loads hand-curated metadata for the sample proposals in
``data/proposals/template/`` so the overview page + Africa map work today.
Later this is replaced by the real ingestion pipeline (see
``docs/proposal_extraction_implementation_plan.md``); the page only depends on
the two DataFrame shapes returned here, so swapping the source is a drop-in.

``load_proposals()``    → one row per proposal (metadata for the overview).
``proposal_sites()``    → one row per project site (lat/lon for the map markers).
``covered_countries()`` → tidy country table for the choropleth highlight.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

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


@st.cache_data
def _raw(path=None) -> list:
    p = Path(path) if path else DATA_PATH
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_proposals(path=None) -> pd.DataFrame:
    """One row per proposal with a human-readable ``location`` summary."""
    rows = []
    for p in _raw(path):
        sites = p.get("sites", [])
        location = "; ".join(s["name"] for s in sites) if sites else "—"
        rows.append(
            {
                "id": p["id"],
                "title": p["title"],
                "donor": p["donor"],
                "sector": p["sector"],
                "summary": p["summary"],
                "partners": p["partners"],
                "timeframe": p["timeframe"],
                "budget": p["budget"],
                "beneficiaries": p["beneficiaries"],
                "countries": p.get("countries", []),
                "n_sites": len(sites),
                "location": location,
                "source_file": p.get("source_file", ""),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def proposal_sites(path=None) -> pd.DataFrame:
    """One row per project site (exploded), carrying proposal metadata."""
    rows = []
    for p in _raw(path):
        for s in p.get("sites", []):
            rows.append(
                {
                    "id": p["id"],
                    "title": p["title"],
                    "donor": p["donor"],
                    "sector": p["sector"],
                    "budget": p["budget"],
                    "timeframe": p["timeframe"],
                    "partners": p["partners"],
                    "beneficiaries": p["beneficiaries"],
                    "site": s["name"],
                    "lat": s["lat"],
                    "lon": s["lon"],
                }
            )
    return pd.DataFrame(rows)


@st.cache_data
def covered_countries(path=None) -> pd.DataFrame:
    """Country → number of proposals active there (for the choropleth)."""
    counts = {}
    for p in _raw(path):
        for c in p.get("countries", []):
            counts[c] = counts.get(c, 0) + 1
    return (
        pd.DataFrame({"country": list(counts), "proposals": list(counts.values())})
        .sort_values("proposals", ascending=False)
        .reset_index(drop=True)
    )
