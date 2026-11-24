"""Page 10 — Indicator Matching: results of the ESGReveal-style pipeline per document and across the portfolio.

Reads the CSVs in ``outputs/latest/`` (no LLM calls, no pipeline dependencies). Views: filters (country, sector,
match type, document); the per-document three-column table (document indicator | IndiKit indicator (ID) | match)
with evidence quotes on expand and a link to the IndiKit page, plus the closest IndiKit indicator as a suggestion for
rows without a match; coverage (IndiKit indicators by number of documents,
never-used indicators, sector x country heat map); novel indicators and their clusters; documents whose indicators
are in another document; download buttons for the CSVs and the review file.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis_log import log_section
from src.config import INDIKIT_URL
from src.plots import crosstab_heatmap
from src.results import FILES, closest_document_indicators, document_label, file_bytes, has_results, indikit_label, nearest_names, numeric, parse_nearest, parse_nearest_pair, read_text, run_id, table

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
names = nearest_names()
SUGGESTION_FLOOR = 0.40   # below this cosine similarity the closest IndiKit indicator is usually unrelated


def closest_suggestion(row) -> str:
    """Closest IndiKit indicator for a row with no match (name, id, similarity); blank for matched rows."""
    if row["match_type"] != "none":
        return ""
    near = parse_nearest(row.get("nearest_indikit", ""))
    if not near:
        return ""
    ind, score = near[0]
    label = indikit_label(ind, score, names)
    return label if score != score or score >= SUGGESTION_FLOOR else f"{label} (weak)"

# every IndiKit indicator of the library with its sector (used + never used = the whole library)
universe = pd.concat([most_used[["indikit_id", "indikit_name", "sector"]], never_used[["indikit_id", "indikit_name", "sector"]]], ignore_index=True).drop_duplicates("indikit_id") if len(most_used) or len(never_used) else pd.DataFrame(columns=["indikit_id", "indikit_name", "sector"])
id_sector = universe.set_index("indikit_id")["sector"].to_dict()
ALL_SECTORS = "(all sectors)"
SECTOR_OPTIONS = [ALL_SECTORS] + sorted(universe["sector"].unique())


def closest_id(value) -> str:
    near = parse_nearest(value)
    return near[0][0] if near else ""


# --- Filters ---------------------------------------------------------------
# The sector is chosen next to the pie chart on the Coverage tab (widget key "pie-sector"); Streamlit reruns the page on
# every change, so its value is already in session state here and filters everything below, including the KPIs.
sector = st.session_state.get("pie-sector", ALL_SECTORS)
if sector not in SECTOR_OPTIONS:
    sector = ALL_SECTORS
with st.sidebar:
    st.header("Filters")
    countries = st.multiselect("Country", sorted(summary["country_name"].unique()))
    match_types = st.multiselect("Match type", ["verbatim", "close", "none"])
    st.caption(f"IndiKit sector: **{sector}** - change it with the *Sector* box above the pie chart on the Coverage tab. "
               "Unmatched document indicators count for the sector of their closest IndiKit indicator.")

f_summary = summary[summary["country_name"].isin(countries)] if countries else summary
f_docs = set(f_summary["doc_id"])
f_ind = doc_ind[doc_ind["doc_id"].isin(f_docs)]
if match_types:
    f_ind = f_ind[f_ind["match_type"].isin(match_types)]
f_matrix = matrix[matrix["doc_id"].isin(f_docs)]
f_novel = novel[novel["doc_id"].isin(f_docs)]
f_never_used = never_used
if sector != ALL_SECTORS:
    f_matrix = f_matrix[f_matrix["sector"] == sector]
    f_never_used = never_used[never_used["sector"] == sector]
    # matched rows: the matched IndiKit indicator is in the sector; unmatched rows: their closest IndiKit indicator is
    in_sector = f_ind["indikit_id"].map(id_sector).eq(sector) | ((f_ind["match_type"] == "none") & f_ind["nearest_indikit"].map(closest_id).map(id_sector).eq(sector))
    f_ind = f_ind[in_sector]
    f_novel = f_novel[f_novel["nearest_indikit_ids"].map(lambda v: str(v).split(";")[0].strip()).map(id_sector).eq(sector)]

# --- KPIs ------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Documents", len(f_summary))
k2.metric("Document indicators", len(f_ind))
k3.metric("Verbatim", int((f_ind["match_type"] == "verbatim").sum()))
k4.metric("Close", int((f_ind["match_type"] == "close").sum()))
k5.metric("No IndiKit match", int((f_ind["match_type"] == "none").sum()))

tab_doc, tab_cov, tab_novel, tab_else, tab_dl = st.tabs(["Per document", "Coverage", "Not in IndiKit", "Indicators elsewhere", "Downloads"])

# --- Per document ----------------------------------------------------------
MATCH_WORDS = {"verbatim": "verbatim (IndiKit wording used as is)", "close": "close (same measure, own wording)", "none": "not in IndiKit"}
with tab_doc:
    options = f_summary.sort_values("document")["doc_id"].tolist()
    if not options:
        st.info("No document matches the filters.")
    else:
        choice = st.selectbox("Proposal / logframe", options, format_func=lambda d: doc_labels.get(d, d))
        row = f_summary[f_summary["doc_id"] == choice].iloc[0]
        d = f_ind[f_ind["doc_id"] == choice].copy()
        if row.get("note"):
            st.info(f"📎 {row['note']}")
        n_total, n_vb, n_cl = int(row["total_indicators"] or 0), int(row["verbatim"] or 0), int(row["close"] or 0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Indicators in this document", n_total)
        m2.metric("Matched to IndiKit", n_vb + n_cl, help="verbatim + close")
        m3.metric("Of which verbatim", n_vb, help="IndiKit wording used as is")
        m4.metric("Not in IndiKit", max(n_total - n_vb - n_cl, 0))
        if sector != ALL_SECTORS:
            st.caption(f"Showing the {len(d)} indicators that belong to the sector **{sector}** (matched ones by their IndiKit indicator, unmatched ones by the closest IndiKit indicator).")
        d["How it matches"] = d["match_type"].map(MATCH_WORDS).fillna(d["match_type"])
        d["IndiKit indicator"] = d.apply(lambda r: f"{r['indikit_name']} ({r['indikit_id']})" if r["indikit_id"] else "", axis=1)
        d["IndiKit page"] = d["indikit_id"].map(lambda i: f"{INDIKIT_URL}{i}" if i else "")
        d["Closest IndiKit indicator (not a match)"] = d.apply(closest_suggestion, axis=1)
        d["Evidence from the document"] = d["evidence_quote"]
        d["To check"] = d["needs_review"].astype(str).str.lower().eq("true").map({True: "yes", False: ""})
        show = d.rename(columns={"indicator_text": "Indicator as written in the document"})[
            ["Indicator as written in the document", "How it matches", "IndiKit indicator", "Evidence from the document", "To check", "Closest IndiKit indicator (not a match)", "IndiKit page"]]
        st.dataframe(show, width="stretch", hide_index=True, column_config={"IndiKit page": st.column_config.LinkColumn("IndiKit page", display_text="open")})
        st.caption("**How to read this.** *verbatim*: the document uses the IndiKit indicator as is. *close*: it measures the same thing in its own words. "
                   "*Not in IndiKit*: nothing in IndiKit measures the same thing. *Evidence* is the quote the model based its answer on. "
                   "*To check* marks the rows a MEAL colleague should confirm: every close match, and anything the model was unsure about. "
                   "*Closest IndiKit indicator* only appears for unmatched rows and is a hint at the nearest wording, not a match.")
        with st.expander("Details for analysts: values, levels, sources, reasons, alternatives"):
            for _, r in d.iterrows():
                head = f"**{r['doc_indicator_id']}** {r['indicator_text']} — *{r['match_type']}*" + (f" → [{r['indikit_name']} ({r['indikit_id']})]({INDIKIT_URL}{r['indikit_id']})" if r["indikit_id"] else "")
                st.markdown(head)
                bits = [f"baseline: {r['baseline'] or '—'}", f"target: {r['target'] or '—'}", f"level: {r['level']}", f"location: {r['location']}", f"source: {r['source']}"]
                if r.get("alternate_ids"):
                    bits.append(f"alternatives: {r['alternate_ids']}")
                if r.get("needs_review") == "True":
                    bits.append(f"to check because: {r['needs_review_reason']}")
                st.caption(" · ".join(bits))
                if r.get("reason"):
                    st.caption(f"model's reason: {r['reason']}")
                if r["match_type"] == "none":
                    near = parse_nearest(r.get("nearest_indikit", ""))[:5]
                    if near:
                        st.caption("closest IndiKit indicators: " + " · ".join(f"[{indikit_label(i, s, names)}]({INDIKIT_URL}{i})" for i, s in near))

# --- Coverage --------------------------------------------------------------
PRESENCE_ORDER = ["verbatim", "close", "closest mention", "not present"]
PRESENCE_WORDS = {"verbatim": "used as is (verbatim)", "close": "same measure, own wording (close)", "closest mention": "a proposal comes close (not a match)", "not present": "not used"}
PRESENCE_COLORS = {"used as is (verbatim)": "#2a78d6", "same measure, own wording (close)": "#eb6834", "a proposal comes close (not a match)": "#1baf7a", "not used": "#eda100"}
with tab_cov:
    # IndiKit id -> document indicators (of the selected countries) that have it among their closest
    candidates = closest_document_indicators(doc_ind[doc_ind["doc_id"].isin(f_docs)], doc_labels)

    st.subheader("How much of IndiKit do the proposals use?")
    c1, c2, c3 = st.columns([2, 1, 2])
    pie_sector = c1.selectbox("Sector (applies to the whole page)", SECTOR_OPTIONS, key="pie-sector")
    use_mentions = c2.toggle("Show near misses", value=False, key="pie-mentions", help="Also show the unused IndiKit indicators that some proposal indicator comes close to in wording. This is a hint, not a match.")
    mention_floor = c3.slider("How close the wording must be", 0.30, 0.90, 0.60, 0.05, key="pie-floor", disabled=not use_mentions, help="0 = unrelated, 1 = identical wording")
    country_matrix = matrix[matrix["doc_id"].isin(f_docs)]
    best_match = country_matrix.groupby("indikit_id")["match_type"].agg(lambda s: "verbatim" if (s == "verbatim").any() else "close").to_dict()

    def presence(ind: str) -> str:
        if ind in best_match:
            return best_match[ind]
        if use_mentions and candidates.get(ind) and candidates[ind][0][0] >= mention_floor:
            return "closest mention"
        return "not present"

    uni = universe.copy()
    uni["presence"] = uni["indikit_id"].map(presence)
    uni["closest indicator in a proposal"] = uni["indikit_id"].map(lambda i: candidates[i][0][1] if candidates.get(i) else "")
    uni["how close (0-1)"] = uni["indikit_id"].map(lambda i: round(candidates[i][0][0], 2) if candidates.get(i) and candidates[i][0][0] >= 0 else None)
    uni["in proposal"] = uni["indikit_id"].map(lambda i: candidates[i][0][2] if candidates.get(i) else "")
    sel = uni if pie_sector == ALL_SECTORS else uni[uni["sector"] == pie_sector]
    counts = sel["presence"].value_counts().reindex(PRESENCE_ORDER, fill_value=0)
    counts = counts[(counts > 0) | (counts.index != "closest mention")]
    n_used = int(counts.get("verbatim", 0) + counts.get("close", 0))
    if len(sel):
        pc1, pc2 = st.columns([3, 2])
        pie = counts.reset_index(name="indicators").rename(columns={"index": "presence"})
        pie["status"] = pie["presence"].map(PRESENCE_WORDS)
        fig = px.pie(pie, names="status", values="indicators", hole=0.45, color="status", color_discrete_map=PRESENCE_COLORS, category_orders={"status": list(PRESENCE_WORDS.values())},
                     title=f"{pie_sector}: {n_used} of {len(sel)} IndiKit indicators are used in the proposals")
        fig.update_traces(sort=False, textinfo="label+value+percent", textposition="auto", marker=dict(line=dict(color="white", width=2)), hovertemplate="<b>%{label}</b><br>%{value} indicators (%{percent})<extra></extra>")
        fig.update_layout(legend_title_text="", height=420, margin=dict(t=60, b=20))
        pc1.plotly_chart(fig, width="stretch")
        by_sector = uni.groupby(["sector", "presence"]).size().unstack(fill_value=0).reindex(columns=[c for c in PRESENCE_ORDER if c in uni["presence"].unique()], fill_value=0)
        by_sector["used"] = by_sector.get("verbatim", 0) + by_sector.get("close", 0)
        by_sector["total"] = by_sector[[c for c in PRESENCE_ORDER if c in by_sector.columns]].sum(axis=1)
        by_sector["used %"] = (100 * by_sector["used"] / by_sector["total"]).round(0).astype(int)
        pc2.caption("All sectors at a glance")
        pc2.dataframe(by_sector.sort_values("total", ascending=False)[["total", "used", "used %"] + [c for c in ("closest mention",) if c in by_sector.columns]].rename(columns={"closest mention": "near misses"}), width="stretch")
        st.caption("**How to read this.** An IndiKit indicator counts as *used* when at least one proposal in the selection matches it, as is or in its own words. "
                   "*Not used* means no proposal matches it. With *near misses* on, the unused indicators that some proposal indicator resembles are shown separately; that is a hint for the reviewer, not a match, and does not change the counts of used indicators.")
        if use_mentions:
            mentioned = sel[sel["presence"] == "closest mention"].sort_values("how close (0-1)", ascending=False)
            with st.expander(f"Near misses in {pie_sector} ({len(mentioned)}): unused IndiKit indicators that a proposal comes close to"):
                if len(mentioned):
                    st.dataframe(mentioned[["indikit_name", "sector", "closest indicator in a proposal", "how close (0-1)", "in proposal"]].rename(columns={"indikit_name": "IndiKit indicator"}), width="stretch", hide_index=True)
                else:
                    st.info("No unused indicator in this sector has a proposal indicator this close in wording.")
    else:
        st.info("No IndiKit indicators in this selection.")

    st.subheader("Which IndiKit indicators are used most?")
    mu = f_matrix.groupby(["indikit_id", "indikit_name", "sector"]).agg(proposals=("doc_id", "nunique"), verbatim=("match_type", lambda s: int((s == "verbatim").sum())), close=("match_type", lambda s: int((s == "close").sum()))).reset_index().sort_values("proposals", ascending=False)
    if len(mu):
        top = mu.head(20).sort_values("proposals")
        fig = px.bar(top, x="proposals", y=top.apply(lambda r: f"{r['indikit_name']} ({r['indikit_id']})", axis=1), orientation="h", color="sector", title="The 20 most-used IndiKit indicators (number of proposals)")
        fig.update_layout(yaxis_title="", xaxis_title="proposals", legend_title_text="", height=max(400, 26 * len(top)))
        st.plotly_chart(fig, width="stretch")
        with st.expander("Full list"):
            st.dataframe(mu.rename(columns={"indikit_id": "ID", "indikit_name": "IndiKit indicator"}), width="stretch", hide_index=True)
    else:
        st.info("No matches in the current selection.")

    st.subheader("Which IndiKit indicators does nobody use?")
    st.caption(f"{len(f_never_used)} IndiKit indicators" + (f" in {sector}" if sector != ALL_SECTORS else "") + " are matched by no proposal. "
               "For each, the table shows the proposal indicator that comes closest in wording, as a hint at where the measure may be hiding under another name. "
               "The page *Analysis IndiKit* goes deeper into this list.")
    nu = f_never_used.copy()
    top = nu["indikit_id"].map(lambda i: candidates.get(i, [None])[0])
    nu["closest indicator in a proposal"] = top.map(lambda t: t[1] if t else "")
    nu["how close (0-1)"] = top.map(lambda t: round(t[0], 2) if t and t[0] >= 0 else None)
    nu["in proposal"] = top.map(lambda t: t[2] if t else "")
    nu["proposals that come close"] = nu["indikit_id"].map(lambda i: len(candidates.get(i, [])))
    nu = nu.sort_values(["how close (0-1)", "indikit_name"], ascending=[False, True], na_position="last")
    st.dataframe(nu.rename(columns={"indikit_id": "ID", "indikit_name": "IndiKit indicator"})[["ID", "IndiKit indicator", "sector", "closest indicator in a proposal", "how close (0-1)", "in proposal", "proposals that come close"]], width="stretch", hide_index=True)
    with_cands = nu[nu["proposals that come close"] > 0]
    if len(with_cands):
        with st.expander("See every proposal indicator that comes close to one unused IndiKit indicator"):
            pick_nu = st.selectbox("IndiKit indicator", with_cands["indikit_id"].tolist(), format_func=lambda i: indikit_label(i, None, names), key="never-used-pick")
            st.dataframe(pd.DataFrame([{"how close (0-1)": round(sc, 2) if sc >= 0 else None, "proposal indicator": txt, "proposal": doc} for sc, txt, doc, did in candidates[pick_nu]]),
                         width="stretch", hide_index=True)
    st.subheader("Where are the matches? Sector by country")
    if len(f_matrix):
        ct = pd.crosstab(f_matrix["sector"], f_matrix["country_name"])
        st.plotly_chart(crosstab_heatmap(ct, "Matched IndiKit indicators per sector and country"), width="stretch")

# --- Not in IndiKit --------------------------------------------------------
with tab_novel:
    n = f_novel.copy()
    n["document"] = n["doc_id"].map(doc_labels)
    n["content"] = n["doc_id"].str[-8:]           # identical files in several country folders share this suffix
    st.subheader("Indicators the proposals use that IndiKit does not have")
    st.caption(f"{len(n)} proposal indicators matched nothing in IndiKit" + (f" (closest IndiKit indicator in {sector})" if sector != ALL_SECTORS else "")
               + ". About half are counts of activities (trainings held, sessions organised, people reached) rather than indicators. "
               "The table below groups the same indicator worded differently across proposals; the ones used in several proposals or countries are candidates to add to IndiKit.")
    if len(n) and "cluster_id" in n.columns:
        grp = n.groupby("cluster_id").agg(**{"indicator (shortest wording)": ("indicator_text", lambda x: min(x, key=len)), "proposals": ("content", "nunique"), "documents": ("doc_id", "nunique"),
                                              "countries": ("country_name", lambda x: ", ".join(sorted(set(x))))}).reset_index()
        grp = grp.sort_values(["proposals", "documents"], ascending=False)
        only_several = st.checkbox("Only indicators used in more than one proposal", value=True, key="novel-several")
        shown = grp[grp["proposals"] > 1] if only_several else grp
        st.caption(f"{len(shown)} groups. 'proposals' counts distinct files (a proposal filed in several country folders counts once); 'documents' counts every copy.")
        st.dataframe(shown[["indicator (shortest wording)", "proposals", "documents", "countries"]], width="stretch", hide_index=True)
        pick = st.selectbox("Show all wordings and documents of", ["(choose one)"] + shown["cluster_id"].tolist(), format_func=lambda c: "(choose one)" if c == "(choose one)" else grp.set_index("cluster_id").loc[c, "indicator (shortest wording)"][:90], key="novel-pick")
        if pick != "(choose one)":
            m = n[n["cluster_id"] == pick]
            st.dataframe(m[["document", "country_name", "indicator_text", "level", "baseline", "target"]], width="stretch", hide_index=True)
    with st.expander("Full list with the closest IndiKit indicators (for analysts)"):
        near_ids = n["nearest_indikit_ids"] if "nearest_indikit_ids" in n.columns else [""] * len(n)
        near_sims = n["nearest_similarities"] if "nearest_similarities" in n.columns else [""] * len(n)
        n["Closest IndiKit indicators"] = [" · ".join(indikit_label(i, s, names) for i, s in parse_nearest_pair(ids, sims)[:3]) for ids, sims in zip(near_ids, near_sims)]
        st.caption("Each row is one proposal indicator with no IndiKit match; the last column names the three IndiKit indicators closest in wording, with their similarity.")
        cols = [c for c in ["document", "country_name", "indicator_text", "level", "baseline", "target", "Closest IndiKit indicators", "cluster_id"] if c in n.columns]
        st.dataframe(n[cols], width="stretch", hide_index=True)

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
            tables={"per_document_summary": f_summary, "disclosure_matrix": f_matrix}, context={"run": run_id(), "countries": countries or "all", "sector": sector, "match_types": match_types or "all"})
