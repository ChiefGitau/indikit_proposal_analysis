"""Page 11 — Analysis IndiKit: which of the IndiKit indicators on the live website (zoa.indikit.net) the proposals never
use, presented in three layers. Indicators that exist only in the downloaded Excel export (``on_site = false``) are
excluded from every count; a note states how many were left out and how many of those the proposals use anyway.

1. Per sector: used vs never used (stacked bars).
2. Per sub-sector: the share never used, with the sub-sectors nobody uses at all called out.
3. The unused indicators themselves, in two groups: "there under another name" (a proposal indicator comes close in
   wording) or "not measured by any proposal".

Reads ``outputs/latest/`` and the IndiKit metadata library; no LLM calls. Everything can be downloaded as one Excel
file (one sheet per layer / group) or as CSV.
"""

from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis_log import log_section
from src.config import INDIKIT_URL
from src.results import closest_document_indicators, document_label, has_results, library, run_id, table

st.set_page_config(page_title="Analysis IndiKit · ZOA", layout="wide")
st.title("Analysis IndiKit")
st.caption("Which indicators on the IndiKit website does no proposal use, and is the measure perhaps there under another name?")

if not has_results():
    st.warning("⚠️ **No pipeline results yet.** This page reads `outputs/latest/` of the pipeline repository. "
               "Run `python -m pipeline.run run --all --confirm-full-run` (or a pilot with `--pilot --copy-latest`) to populate it.")
    st.stop()

USED, NEVER = "#2a78d6", "#eb6834"   # categorical slots 1 and 2 (validated pair)

# --- Data ------------------------------------------------------------------
full_lib = library()
lib = full_lib[full_lib["on_site"]].copy()               # live website only; export-only indicators are left out
export_only = full_lib[~full_lib["on_site"]]
summary = table("document_summary")
doc_ind = table("document_indicators")
matrix = table("disclosure_matrix")
if "region" in summary.columns:
    summary = summary[summary["region"] != "template"]
summary["document"] = summary.apply(document_label, axis=1)
doc_labels = summary.set_index("doc_id")["document"].to_dict()
doc_ind = doc_ind[doc_ind["doc_id"].isin(doc_labels)]
if "is_indicator" in doc_ind.columns:
    doc_ind = doc_ind[doc_ind["is_indicator"].astype(str).str.lower() != "false"]
matrix = matrix[matrix["doc_id"].isin(doc_labels)]

# --- Controls --------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    countries = st.multiselect("Country", sorted(summary["country_name"].unique()) if "country_name" in summary.columns else [])
    sector_pick = st.multiselect("IndiKit sector", sorted(lib["sector"].unique()))
    floor = st.slider("How close the wording must be to count as 'there under another name'", 0.30, 0.90, 0.60, 0.05,
                      help="0 = unrelated, 1 = identical wording. An unused IndiKit indicator counts as 'there under another name' when some proposal indicator is at least this close to it.")

f_summary = summary[summary["country_name"].isin(countries)] if countries else summary
f_docs = set(f_summary["doc_id"])
f_matrix = matrix[matrix["doc_id"].isin(f_docs)]
candidates = closest_document_indicators(doc_ind[doc_ind["doc_id"].isin(f_docs)], doc_labels)

used_docs = f_matrix.groupby("indikit_id")["doc_id"].nunique().to_dict()
best_type = f_matrix.groupby("indikit_id")["match_type"].agg(lambda s: "verbatim" if (s == "verbatim").any() else "close").to_dict()

lib = lib.copy()
lib["documents using it"] = lib["indikit_id"].map(used_docs).fillna(0).astype(int)
lib["used"] = lib["documents using it"] > 0
lib["best match"] = lib["indikit_id"].map(best_type).fillna("")
lib["closest indicator in a proposal"] = lib["indikit_id"].map(lambda i: candidates[i][0][1] if candidates.get(i) else "")
lib["in proposal"] = lib["indikit_id"].map(lambda i: candidates[i][0][2] if candidates.get(i) else "")
lib["how close (0-1)"] = lib["indikit_id"].map(lambda i: round(candidates[i][0][0], 2) if candidates.get(i) and candidates[i][0][0] >= 0 else None)
lib["proposals that come close"] = lib["indikit_id"].map(lambda i: len(candidates.get(i, [])))
lib["IndiKit link"] = lib.apply(lambda r: r["url"] or (f"{INDIKIT_URL}{r['indikit_id']}" if r["on_site"] else ""), axis=1)


def group_of(r) -> str:
    if r["used"]:
        return "used"
    if r["how close (0-1)"] is not None and r["how close (0-1)"] == r["how close (0-1)"] and r["how close (0-1)"] >= floor:
        return "there under another name"
    return "not measured by any proposal"


lib["group"] = lib.apply(group_of, axis=1)
view = lib[lib["sector"].isin(sector_pick)] if sector_pick else lib
never = view[~view["used"]]

# --- Caveats (once, at the top) -------------------------------------------
n_docs = len(f_summary)
n_logframe = int((pd.to_numeric(f_summary.get("total_indicators", 0), errors="coerce").fillna(0) > 0).sum()) if len(f_summary) else 0
dup_groups = f_summary["doc_id"].str[-8:].value_counts() if len(f_summary) else pd.Series(dtype=int)
n_dup_copies = int((dup_groups[dup_groups > 1] - 1).sum()) if len(dup_groups) else 0
n_export_used = int(export_only["indikit_id"].isin(used_docs).sum())
st.info(f"**Before you read on.** This page looks at the {len(lib)} indicators on the IndiKit website. The {len(export_only)} indicators that only exist in the Excel export are left out "
        f"({n_export_used} of those are used in proposals anyway; the Indicator Matching page shows them). "
        f"'Not used' means no proposal among the {n_docs} selected documents matches the indicator; {n_logframe} of those documents contain indicators, the rest are narrative-only or keep their logframe in an annex. "
        f"The matches come from the model and have not yet been checked by a person" + (f". {n_dup_copies} document(s) are copies of another file in the folders" if n_dup_copies else "") + ".")

# --- KPIs ------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Indicators on the website", len(view))
k2.metric("Used by at least one proposal", int(view["used"].sum()))
k3.metric("Not used by any proposal", len(never))
k4.metric("…but there under another name", int((never["group"] == "there under another name").sum()), help="not matched, but a proposal indicator comes close in wording (see the slider)")
k5.metric("…not measured at all", int((never["group"] == "not measured by any proposal").sum()), help="not matched and nothing in any proposal comes close")

tab_sector, tab_sub, tab_list, tab_dl = st.tabs(["By sector", "By sub-sector", "The unused indicators", "Downloads"])

# --- 1. Per sector ---------------------------------------------------------
with tab_sector:
    by_sector = view.groupby("sector").agg(total=("indikit_id", "size"), used=("used", "sum")).reset_index()
    by_sector["never used"] = by_sector["total"] - by_sector["used"]
    by_sector["share never used %"] = (100 * by_sector["never used"] / by_sector["total"]).round(1)
    by_sector = by_sector.sort_values("total", ascending=True)
    st.subheader("How many indicators of each sector do the proposals use?")
    st.caption("Each bar is one IndiKit sector. Blue: used by at least one proposal. Orange: used by none.")
    fig = go.Figure()
    fig.add_bar(y=by_sector["sector"], x=by_sector["used"], name="used", orientation="h", marker_color=USED, marker_line=dict(color="white", width=2),
                text=by_sector["used"], textposition="inside", hovertemplate="<b>%{y}</b><br>used: %{x}<extra></extra>")
    fig.add_bar(y=by_sector["sector"], x=by_sector["never used"], name="not used", orientation="h", marker_color=NEVER, marker_line=dict(color="white", width=2),
                text=by_sector["never used"], textposition="inside", hovertemplate="<b>%{y}</b><br>never used: %{x}<extra></extra>")
    fig.update_layout(barmode="stack", height=max(320, 42 * len(by_sector) + 80), xaxis_title="number of IndiKit indicators", yaxis_title="", legend_title_text="",
                      legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=30, b=30), plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(by_sector.sort_values("total", ascending=False)[["sector", "total", "used", "never used", "share never used %"]].rename(columns={"never used": "not used", "share never used %": "not used %"}), width="stretch", hide_index=True)

# --- 2. Per sub-sector -----------------------------------------------------
with tab_sub:
    by_sub = view.groupby(["sector", "sub_sector"]).agg(total=("indikit_id", "size"), used=("used", "sum")).reset_index()
    by_sub["never used"] = by_sub["total"] - by_sub["used"]
    by_sub["share never used %"] = (100 * by_sub["never used"] / by_sub["total"]).round(1)
    by_sub["label"] = by_sub["sector"].str.replace("cross-cutting: ", "") + " › " + by_sub["sub_sector"]
    unused_subs = by_sub[by_sub["used"] == 0]
    st.subheader("Which sub-sectors are left aside?")
    st.caption("The longer the bar, the larger the part of the sub-sector that no proposal uses. Sub-sectors at 100 % have no indicator in use anywhere.")
    if len(unused_subs):
        st.markdown("**Sub-sectors where no indicator is used at all:** " + " · ".join(f"{r['label']} ({r['total']})" for _, r in unused_subs.sort_values("total", ascending=False).iterrows()))
    plot = by_sub.sort_values(["share never used %", "total"], ascending=[True, True])
    fig2 = go.Figure(go.Bar(y=plot["label"], x=plot["share never used %"], orientation="h", marker_color=NEVER, marker_line=dict(color="white", width=1),
                            text=[f"{nu} of {t}" for nu, t in zip(plot["never used"], plot["total"])], textposition="outside", cliponaxis=False,
                            hovertemplate="<b>%{y}</b><br>%{x}% never used<extra></extra>"))
    fig2.update_layout(height=max(360, 24 * len(plot) + 80), xaxis_title="% of the sub-sector's indicators that no proposal uses", yaxis_title="", xaxis_range=[0, 112],
                       margin=dict(l=10, r=40, t=20, b=30), plot_bgcolor="white", showlegend=False)
    fig2.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    st.plotly_chart(fig2, width="stretch")
    st.dataframe(by_sub.sort_values(["share never used %", "total"], ascending=[False, False])[["sector", "sub_sector", "total", "used", "never used", "share never used %"]].rename(columns={"sub_sector": "sub-sector", "never used": "not used", "share never used %": "not used %"}), width="stretch", hide_index=True)

# --- 3. The unused indicators ----------------------------------------------
GROUPS = [
    ("there under another name", "No proposal matches the IndiKit indicator, but a proposal indicator comes close in wording. The measure seems to be in use; clearer IndiKit wording or guidance to proposal writers could bring it in line. The closest proposal indicator is shown as the evidence."),
    ("not measured by any proposal", "Nothing in any proposal comes close. Either outside what ZOA currently programmes, or an outcome measure that proposals do not commit to."),
]
LIST_COLS = ["indikit_id", "indikit_name", "sector", "sub_sector", "closest indicator in a proposal", "how close (0-1)", "in proposal", "proposals that come close", "IndiKit link"]
NICE = {"indikit_id": "ID", "indikit_name": "IndiKit indicator", "sub_sector": "sub-sector"}
with tab_list:
    st.subheader(f"{len(never)} IndiKit indicators that no proposal uses, in two groups")
    st.caption("Open a group to see its indicators. Rows are sorted so that the ones with the closest proposal indicator come first.")
    for name, blurb in GROUPS:
        g = never[never["group"] == name].sort_values(["how close (0-1)", "sector", "indikit_name"], ascending=[False, True, True], na_position="last")
        with st.expander(f"{name} ({len(g)})", expanded=(name == GROUPS[0][0])):
            st.caption(blurb)
            if len(g):
                st.dataframe(g[LIST_COLS].rename(columns=NICE), width="stretch", hide_index=True, column_config={"IndiKit link": st.column_config.LinkColumn("IndiKit page", display_text="open")})
            else:
                st.info("None in the current selection.")
    with st.expander("See every proposal indicator that comes close to one unused IndiKit indicator"):
        opts = never[never["proposals that come close"] > 0].sort_values("indikit_name")
        if len(opts):
            pick = st.selectbox("IndiKit indicator", opts["indikit_id"].tolist(), format_func=lambda i: f"{opts.set_index('indikit_id').loc[i, 'indikit_name']} ({i})", key="analysis-pick")
            st.dataframe(pd.DataFrame([{"how close (0-1)": round(sc, 2) if sc >= 0 else None, "proposal indicator": txt, "proposal": doc} for sc, txt, doc, did in candidates[pick]]),
                         width="stretch", hide_index=True)
        else:
            st.info("No unused indicator in the selection has a proposal indicator that comes close.")

# --- Downloads -------------------------------------------------------------
with tab_dl:
    st.caption("One Excel file with a sheet per view and per group, or the list of unused indicators as CSV. The filters and the slider on the left apply.")
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"item": ["run", "scope", "countries", "sectors", "similarity floor", "documents in selection", "IndiKit indicators on the website", "used", "not used", "export-only indicators (excluded)", "of which used in proposals"],
                      "value": [run_id(), "indicators on the live IndiKit website only", ", ".join(countries) or "all", ", ".join(sector_pick) or "all", floor, n_docs, len(view), int(view["used"].sum()), len(never), len(export_only), n_export_used]}).to_excel(xw, sheet_name="summary", index=False)
        by_sector.sort_values("total", ascending=False)[["sector", "total", "used", "never used", "share never used %"]].rename(columns={"never used": "not used", "share never used %": "not used %"}).to_excel(xw, sheet_name="by sector", index=False)
        by_sub.sort_values(["share never used %", "total"], ascending=[False, False])[["sector", "sub_sector", "total", "used", "never used", "share never used %"]].rename(columns={"sub_sector": "sub-sector", "never used": "not used", "share never used %": "not used %"}).to_excel(xw, sheet_name="by sub-sector", index=False)
        for name, _ in GROUPS:
            never[never["group"] == name].sort_values(["how close (0-1)", "sector"], ascending=[False, True], na_position="last")[LIST_COLS].rename(columns=NICE).to_excel(xw, sheet_name=name[:31], index=False)
        view.sort_values(["sector", "sub_sector", "indikit_name"])[["indikit_id", "indikit_name", "sector", "sub_sector", "used", "best match", "documents using it", "group"] + LIST_COLS[4:]].rename(columns={**NICE, "documents using it": "proposals using it", "group": "status"}).to_excel(xw, sheet_name="all indicators", index=False)
    st.download_button("indikit_analysis.xlsx", buf.getvalue(), file_name="indikit_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl-analysis-xlsx")
    st.download_button("unused_indikit_indicators.csv", never[["group"] + LIST_COLS].rename(columns={**NICE, "group": "status"}).to_csv(index=False).encode("utf-8"), file_name="unused_indikit_indicators.csv", mime="text/csv", key="dl-analysis-csv")

log_section("Analysis IndiKit", metrics={"indikit_indicators": len(view), "used": int(view["used"].sum()), "never_used": len(never), "there_under_another_name": int((never["group"] == "there under another name").sum()),
                                         "not_measured": int((never["group"] == "not measured by any proposal").sum()), "export_only_excluded": len(export_only), "similarity_floor": floor},
            tables={"per_sector": by_sector, "per_sub_sector": by_sub}, context={"run": run_id(), "countries": countries or "all", "sectors": sector_pick or "all"})
