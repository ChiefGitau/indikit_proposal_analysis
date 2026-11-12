"""Page 10 — Indicator Matching: results of the ESGReveal-style pipeline per document and across the portfolio.

Reads the CSVs in ``outputs/latest/`` (no LLM calls, no pipeline dependencies). Views: filters (country, sector,
match type, document); the per-document three-column table (document indicator | IndiKit indicator (ID) | match)
with evidence quotes on expand and a link to the IndiKit page; coverage (IndiKit indicators by number of documents,
never-used indicators, sector x country heat map); novel indicators and their clusters; documents whose indicators
are in another document; download buttons for the CSVs and the review file.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis_log import log_section
from src.config import INDIKIT_URL
from src.plots import crosstab_heatmap
from src.results import FILES, document_label, file_bytes, has_results, numeric, read_text, run_id, table

st.set_page_config(page_title="Indicator Matching · ZOA", layout="wide")
st.title("Indicator Matching")

if not has_results():
    st.warning("⚠️ **No pipeline results yet.** This page reads `outputs/latest/` of the pipeline repository. "
               "Run `python -m pipeline.run run --all --confirm-full-run` (or a pilot with `--pilot --copy-latest`) to populate it. Nothing is shown in demo mode.")
    st.stop()

st.caption(f"Run `{run_id()}` · which ZOA IndiKit indicators each document contains (verbatim / close), the document indicators with no IndiKit match, and where indicators live in another document.")

summary = numeric(table("document_summary"), ["total_indicators", "verbatim", "close", "none", "pct_matched"])
doc_ind = table("document_indicators")
matrix = table("disclosure_matrix")
novel = table("novel_indicators")
most_used = table("portfolio_most_used_indikit")
never_used = table("portfolio_never_used_indikit")
clusters = table("portfolio_novel_clusters")

summary = summary[summary.get("region", "") != "template"] if "region" in summary.columns else summary
summary["document"] = summary.apply(document_label, axis=1)
doc_labels = summary.set_index("doc_id")["document"].to_dict()
doc_ind = doc_ind[doc_ind["doc_id"].isin(doc_labels)]
matrix = matrix[matrix["doc_id"].isin(doc_labels)]
novel = novel[novel["doc_id"].isin(doc_labels)]
doc_ind = doc_ind[doc_ind["is_indicator"].astype(str).str.lower() != "false"]

# --- Filters ---------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    countries = st.multiselect("Country", sorted(summary["country_name"].unique()))
    sectors = st.multiselect("IndiKit sector", sorted(matrix["sector"].unique()) if len(matrix) else [])
    match_types = st.multiselect("Match type", ["verbatim", "close", "none"])

f_summary = summary[summary["country_name"].isin(countries)] if countries else summary
f_docs = set(f_summary["doc_id"])
f_ind = doc_ind[doc_ind["doc_id"].isin(f_docs)]
if match_types:
    f_ind = f_ind[f_ind["match_type"].isin(match_types)]
f_matrix = matrix[matrix["doc_id"].isin(f_docs)]
if sectors:
    f_matrix = f_matrix[f_matrix["sector"].isin(sectors)]
    sector_ids = set(f_matrix["indikit_id"])
    f_ind = f_ind[f_ind["indikit_id"].isin(sector_ids) | (f_ind["match_type"] == "none")] if match_types != ["none"] else f_ind

# --- KPIs ------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Documents", len(f_summary))
k2.metric("Document indicators", len(f_ind))
k3.metric("Verbatim", int((f_ind["match_type"] == "verbatim").sum()))
k4.metric("Close", int((f_ind["match_type"] == "close").sum()))
k5.metric("No IndiKit match", int((f_ind["match_type"] == "none").sum()))

tab_doc, tab_cov, tab_novel, tab_else, tab_dl = st.tabs(["Per document", "Coverage", "Novel indicators", "Indicators elsewhere", "Downloads"])

# --- Per document ----------------------------------------------------------
with tab_doc:
    options = f_summary.sort_values("document")["doc_id"].tolist()
    if not options:
        st.info("No document matches the filters.")
    else:
        choice = st.selectbox("Document", options, format_func=lambda d: doc_labels.get(d, d))
        row = f_summary[f_summary["doc_id"] == choice].iloc[0]
        d = f_ind[f_ind["doc_id"] == choice].copy()
        if row.get("note"):
            st.info(f"📎 {row['note']}")
        st.markdown(f"**Indicators in document:** {int(row['total_indicators'] or 0)} · **verbatim:** {int(row['verbatim'] or 0)} · **close:** {int(row['close'] or 0)} · "
                    f"**total matched:** {int((row['verbatim'] or 0) + (row['close'] or 0))}" + (f" · referenced document: {row['referenced_documents']}" if row.get("referenced_documents") else ""))
        d["IndiKit indicator (ID)"] = d.apply(lambda r: f"{r['indikit_name']} ({r['indikit_id']})" if r["indikit_id"] else "—", axis=1)
        d["IndiKit link"] = d["indikit_id"].map(lambda i: f"{INDIKIT_URL}{i}" if i else "")
        show = d.rename(columns={"indicator_text": "Indicator in document", "match_type": "Match"})[["doc_indicator_id", "Indicator in document", "IndiKit indicator (ID)", "Match", "needs_review", "IndiKit link"]]
        st.dataframe(show, width="stretch", hide_index=True, column_config={"IndiKit link": st.column_config.LinkColumn("IndiKit page", display_text="open")})
        with st.expander("Evidence, values and reasons"):
            for _, r in d.iterrows():
                head = f"**{r['doc_indicator_id']}** {r['indicator_text']} — *{r['match_type']}*" + (f" → [{r['indikit_name']} ({r['indikit_id']})]({INDIKIT_URL}{r['indikit_id']})" if r["indikit_id"] else "")
                st.markdown(head)
                bits = [f"baseline: {r['baseline'] or '—'}", f"target: {r['target'] or '—'}", f"level: {r['level']}", f"location: {r['location']}", f"source: {r['source']}"]
                if r.get("alternate_ids"):
                    bits.append(f"alternates: {r['alternate_ids']}")
                if r.get("needs_review") == "True":
                    bits.append(f"needs review: {r['needs_review_reason']}")
                st.caption(" · ".join(bits))
                if r.get("evidence_quote"):
                    st.code(r["evidence_quote"], language=None)
                if r.get("reason"):
                    st.caption(f"reason: {r['reason']}")

# --- Coverage --------------------------------------------------------------
with tab_cov:
    st.subheader("IndiKit indicators by number of documents using them")
    mu = f_matrix.groupby(["indikit_id", "indikit_name", "sector"]).agg(documents=("doc_id", "nunique"), verbatim=("match_type", lambda s: int((s == "verbatim").sum()))).reset_index().sort_values("documents", ascending=False)
    if len(mu):
        top = mu.head(30)
        fig = px.bar(top.sort_values("documents"), x="documents", y=top.sort_values("documents").apply(lambda r: f"{r['indikit_name']} ({r['indikit_id']})", axis=1), orientation="h", color="sector", title="Most-used IndiKit indicators (top 30)")
        fig.update_layout(yaxis_title="", height=max(400, 22 * len(top)))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(mu, width="stretch", hide_index=True)
    else:
        st.info("No matches in the current selection.")
    st.subheader("IndiKit indicators never used")
    st.caption(f"{len(never_used)} of the IndiKit indicators were not matched by any document in the run.")
    st.dataframe(never_used, width="stretch", hide_index=True)
    st.subheader("Sector × country")
    if len(f_matrix):
        ct = pd.crosstab(f_matrix["sector"], f_matrix["country_name"])
        st.plotly_chart(crosstab_heatmap(ct, "Matched IndiKit indicators per sector and country"), width="stretch")

# --- Novel -----------------------------------------------------------------
with tab_novel:
    n = novel[novel["doc_id"].isin(f_docs)].copy()
    st.caption(f"{len(n)} document indicators with no IndiKit match (after the reverse check). Clusters group similar wordings across documents: candidates to add to IndiKit.")
    if len(clusters):
        c = clusters.copy()
        c = c[c["cluster_id"].isin(n["cluster_id"].unique())] if len(n) else c
        st.subheader("Clusters")
        st.dataframe(c, width="stretch", hide_index=True)
        pick = st.selectbox("Show the indicators of cluster", ["(all)"] + c["cluster_id"].tolist())
        if pick != "(all)":
            n = n[n["cluster_id"] == pick]
    n["document"] = n["doc_id"].map(doc_labels)
    st.dataframe(n[["document", "doc_indicator_id", "indicator_text", "level", "baseline", "target", "nearest_indikit_ids", "nearest_similarities", "cluster_id"]], width="stretch", hide_index=True)

# --- Elsewhere -------------------------------------------------------------
with tab_else:
    e = f_summary[(f_summary["indicators_elsewhere"].astype(str).str.lower() == "true") | (f_summary["indicators_also_elsewhere"].astype(str).str.lower() == "true")]
    st.caption("Documents that point to another document for their indicators (logframe annex, indicator tracker, IPTT). The document names come from the reference detection.")
    st.dataframe(e[["document", "country_name", "indicators_elsewhere", "indicators_also_elsewhere", "referenced_documents", "note"]], width="stretch", hide_index=True)

# --- Downloads -------------------------------------------------------------
with tab_dl:
    for key, label in (("document_indicators", "document_indicators.csv"), ("disclosure_matrix", "disclosure_matrix.csv"), ("novel_indicators", "novel_indicators.csv"), ("document_summary", "document_summary.csv"),
                       ("portfolio_most_used_indikit", "portfolio_most_used_indikit.csv"), ("portfolio_never_used_indikit", "portfolio_never_used_indikit.csv"), ("portfolio_novel_clusters", "portfolio_novel_clusters.csv")):
        data = file_bytes(key)
        if data:
            st.download_button(label, data, file_name=label, mime="text/csv", key=f"dl-{key}")
    rev = file_bytes("review")
    if rev:
        st.download_button("review.xlsx (rows that need review)", rev, file_name="review.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl-review")
    else:
        st.caption("No review file yet: run `python -m pipeline.run export-review`.")
    with st.expander("Portfolio summary (Markdown)"):
        st.markdown(read_text("portfolio_summary") or "_no summary_")
    with st.expander("Run report"):
        st.markdown(read_text("run_report") or "_no report_")

log_section("Indicator Matching", metrics={"documents": len(f_summary), "document_indicators": len(f_ind), "verbatim": int((f_ind["match_type"] == "verbatim").sum()), "close": int((f_ind["match_type"] == "close").sum()), "none": int((f_ind["match_type"] == "none").sum())},
            tables={"per_document_summary": f_summary, "disclosure_matrix": f_matrix}, context={"run": run_id(), "countries": countries or "all", "sectors": sectors or "all", "match_types": match_types or "all"})
