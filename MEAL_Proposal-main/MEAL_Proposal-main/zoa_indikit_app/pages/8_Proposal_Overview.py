"""Page 8 — Proposal Overview: what we can read off each proposal + Africa map.

DUMMY data for now (hand-curated from the five sample proposals in
``data/proposals/template/``). The page only depends on the DataFrame shapes
returned by ``src.proposals``, so the real ingestion pipeline can replace the
source later without touching this page.
"""

import streamlit as st

from src.analysis_log import log_section
from src.plots import africa_project_map
from src.proposals import OVERVIEW_COLS, covered_countries, load_proposals, proposal_sites

st.set_page_config(page_title="Proposal Overview · ZOA", layout="wide")
st.title("Proposal Overview")
st.caption(
    "General information extracted per proposal — location, donor, budget, "
    "timeframe, partners and summary — plus an Africa map of where ZOA works. "
    "⚠️ **Dummy data** curated from the five sample proposals; the document "
    "ingestion pipeline will populate this automatically later."
)

props = load_proposals()
sites = proposal_sites()
countries = covered_countries()

# --- KPI row ---------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Proposals", len(props))
k2.metric("Countries", countries["country"].nunique())
k3.metric("Project sites", len(sites))
k4.metric("Donors", props["donor"].nunique())

# --- Africa map ------------------------------------------------------------
st.subheader("Where ZOA works")
st.caption("Shaded countries have at least one proposal · hover a marker for the details.")
st.plotly_chart(
    africa_project_map(countries, sites),
    width="stretch",
    config={"scrollZoom": True},
)

# --- Per-proposal detail ---------------------------------------------------
st.subheader("Proposal details")
choice = st.selectbox(
    "Inspect a proposal",
    options=props["id"],
    format_func=lambda i: f"{i} — {props.loc[props['id'] == i, 'title'].iloc[0]}",
)
row = props.loc[props["id"] == choice].iloc[0]

c1, c2, c3 = st.columns(3)
c1.markdown(f"**Donor**\n\n{row.donor}")
c1.markdown(f"**Sector**\n\n{row.sector}")
c2.markdown(f"**Location**\n\n{row.location}")
c2.markdown(f"**Timeframe**\n\n{row.timeframe}")
c3.markdown(f"**Budget**\n\n{row.budget}")
c3.markdown(f"**Partners**\n\n{row.partners}")
st.markdown(f"**Beneficiaries** · {row.beneficiaries}")
st.markdown(f"**Summary** · {row.summary}")
st.caption(f"Source file: `{row.source_file}`")

# --- All proposals table ---------------------------------------------------
st.subheader("All proposals")
st.dataframe(props[OVERVIEW_COLS], width="stretch", hide_index=True)

# --- Persist to the analysis log -------------------------------------------
log_section(
    "Proposal Overview",
    metrics={
        "proposals": len(props),
        "countries": int(countries["country"].nunique()),
        "project_sites": len(sites),
        "donors": int(props["donor"].nunique()),
    },
    tables={"proposals": props[OVERVIEW_COLS], "sites": sites, "countries": countries},
    context={"source": "dummy (data/proposals_dummy.json)"},
)
