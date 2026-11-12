"""Page 8 — Proposal Overview: per-document results of the indicator pipeline + map of where the documents come from.

Reads ``outputs/latest/document_summary.csv`` through ``src.proposals``; when no pipeline run exists the page falls
back to the hand-curated dummy file and says so in a banner.
"""

import pandas as pd
import streamlit as st

from src.analysis_log import log_section
from src.plots import africa_project_map
from src.proposals import OVERVIEW_COLS, RESULT_COLS, covered_countries, is_demo, load_proposals, proposal_sites
from src.results import run_id

st.set_page_config(page_title="Proposal Overview · ZOA", layout="wide")
st.title("Proposal Overview")

demo = is_demo()
if demo:
    st.warning("⚠️ **Demo data.** No pipeline run was found in `outputs/latest/`; showing the five hand-curated sample proposals. "
               "Run `python -m pipeline.run run --all --confirm-full-run` in the pipeline repository to populate this page.")
    st.caption("General information per proposal — location, donor, budget, timeframe, partners and summary — plus a map of where ZOA works.")
else:
    st.caption(f"Per-document results of the IndiKit matching pipeline (run `{run_id()}`): indicators found, share matched to IndiKit, and documents whose indicators live in another file.")

props = load_proposals()
sites = proposal_sites()
countries = covered_countries()

# --- KPI row ---------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Documents" if not demo else "Proposals", len(props))
k2.metric("Countries", countries["country"].nunique())
if demo:
    k3.metric("Project sites", len(sites))
    k4.metric("Donors", props["donor"].nunique())
else:
    tot = pd.to_numeric(props["total_indicators"], errors="coerce").fillna(0)
    matched = pd.to_numeric(props["verbatim"], errors="coerce").fillna(0) + pd.to_numeric(props["close"], errors="coerce").fillna(0)
    k3.metric("Document indicators", int(tot.sum()))
    k4.metric("Share matched to IndiKit", f"{(100 * matched.sum() / tot.sum()):.0f}%" if tot.sum() else "—")

# --- Map -------------------------------------------------------------------
st.subheader("Where the documents come from")
st.caption("Shaded countries have at least one document." + (" Hover a marker for the details." if len(sites) else ""))
scope = "africa" if countries["country"].isin(["Syria", "Yemen"]).sum() == 0 else "world"
st.plotly_chart(africa_project_map(countries, sites, title="Documents per country", scope=scope), width="stretch", config={"scrollZoom": True})

# --- Share matched per country / document --------------------------------
if not demo:
    st.subheader("Share of document indicators matched to IndiKit")
    by_country = props.assign(total=pd.to_numeric(props["total_indicators"], errors="coerce").fillna(0), matched=matched).groupby("country_name")[["total", "matched"]].sum()
    by_country["share matched %"] = (100 * by_country["matched"] / by_country["total"].replace(0, pd.NA)).round(1)
    by_country["documents"] = props.groupby("country_name").size()
    st.dataframe(by_country.reset_index().sort_values("documents", ascending=False), width="stretch", hide_index=True)

# --- Per-proposal detail ---------------------------------------------------
st.subheader("Document details" if not demo else "Proposal details")
choice = st.selectbox("Inspect a document", options=props["id"], format_func=lambda i: f"{props.loc[props['id'] == i, 'title'].iloc[0]}")
row = props.loc[props["id"] == choice].iloc[0]

if demo:
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Donor**\n\n{row.donor}")
    c1.markdown(f"**Sector**\n\n{row.sector}")
    c2.markdown(f"**Location**\n\n{row.location}")
    c2.markdown(f"**Timeframe**\n\n{row.timeframe}")
    c3.markdown(f"**Budget**\n\n{row.budget}")
    c3.markdown(f"**Partners**\n\n{row.partners}")
    st.markdown(f"**Beneficiaries** · {row.beneficiaries}")
    st.markdown(f"**Summary** · {row.summary}")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Indicators in document", int(pd.to_numeric(row.total_indicators, errors="coerce") or 0))
    c2.metric("Verbatim / close", f"{int(pd.to_numeric(row.verbatim, errors='coerce') or 0)} / {int(pd.to_numeric(row.close, errors='coerce') or 0)}")
    c3.metric("Share matched", f"{row.pct_matched}%" if str(row.pct_matched) not in ("", "nan", "None") else "—")
    if row.indicators_elsewhere:
        st.info(f"📎 {row.note or 'Indicators are in another document'}")
    elif row.note:
        st.caption(row.note)
    st.caption(f"Country: {row.country_name} · language: {row.language} · parse status: {row.parse_status} · pairs asked: {row.pairs_asked} · LLM calls: {row.llm_calls}")
    st.page_link("pages/10_Indicator_Matching.py", label="→ open the indicator table of this document on the Indicator Matching page")
st.caption(f"Source file: `{row.source_file}`")

# --- All proposals table ---------------------------------------------------
st.subheader("All documents" if not demo else "All proposals")
cols = OVERVIEW_COLS if demo else ["title"] + [c for c in RESULT_COLS if c not in ("doc_id",)]
st.dataframe(props[cols], width="stretch", hide_index=True)

# --- Persist to the analysis log -------------------------------------------
log_section(
    "Proposal Overview",
    metrics={"proposals": len(props), "countries": int(countries["country"].nunique()), "project_sites": len(sites), "donors": int(props["donor"].nunique())},
    tables={"proposals": props[cols], "sites": sites, "countries": countries},
    context={"source": "dummy (data/proposals_dummy.json)" if demo else f"pipeline outputs/latest ({run_id()})"},
)
