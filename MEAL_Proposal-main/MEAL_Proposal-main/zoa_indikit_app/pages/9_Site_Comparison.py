"""Page 9 — Site Comparison: how well does the local CSV match the live site?

Scrapes https://zoa.indikit.net (BeautifulSoup, honouring the 1 req/s crawl
delay) and diffs each scraped indicator page against the ``Export indikit.csv``
snapshot. Surfaces three things:

* **Coverage**   — which CSV indicators still exist on the site (via the sitemap)
                   and which were removed since the export.
* **Field drift** — per-field similarity (name, EN/FR wording, purpose,
                   section/sector/sub-sector) between CSV and live page.
* **Inspector**  — side-by-side CSV vs scraped text for any single indicator.

The crawl is cached to disk, so you only pay the ~5-minute full scrape once;
every rerun reads the cache.

By default the page is read-only: it compares against the scrape committed to the
repository and makes no requests. The scrape and refresh controls only appear when
``ZOA_LOCAL_MODE=1`` is set (see ``src/config.py``).
"""

import subprocess
import sys

import pandas as pd
import streamlit as st

from src.analysis_log import log_section
from src.config import LOCAL_MODE, PIPELINE_ROOT
from src.scraper import (
    BASE_URL,
    build_comparison,
    build_level_comparison,
    crawl,
    fetch_site_levels,
    fetch_sitemap_ids,
    load_cache,
    store_live_ids,
)
from src.state import active_filter_context, get_data, show_active_count

st.set_page_config(page_title="Site Comparison · IndiKit", layout="wide")
st.title("Site Comparison")
st.caption(
    "Scrape the live IndiKit site and compare it to the local CSV export — "
    "what's been removed, and where the text has drifted since the snapshot."
)

full, df = get_data()
show_active_count(df, full)



def refresh_pipeline_data() -> None:
    """Run ``python -m pipeline.run refresh-site`` in the pipeline repository: rebuild the IndiKit metadata from the new
    scrape, rewrite the metadata-dependent output files of the latest run, copy them to outputs/latest and re-export
    Power BI. No LLM call. Then clear the app caches so every page picks the new data up."""
    cmd = [sys.executable, "-m", "pipeline.run", "refresh-site"]
    with st.spinner("Refreshing the pipeline data (metadata, run outputs, Power BI) …"):
        try:
            proc = subprocess.run(cmd, cwd=str(PIPELINE_ROOT), capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.TimeoutExpired) as exc:
            st.error(f"Could not run the refresh: {exc}")
            return
    st.cache_data.clear()
    if proc.returncode == 0:
        st.success("Pipeline data refreshed. The other pages now reflect the current state of the site.")
        with st.expander("What was refreshed"):
            st.code(proc.stdout.strip() or "(no output)", language=None)
    else:
        st.error("The refresh failed; the other pages still show the previous state.")
        st.code((proc.stdout + "\n" + proc.stderr).strip()[-4000:], language=None)
    st.session_state["pipeline_refreshed_at"] = load_cache().get("scraped_at", "")


# --------------------------------------------------------------------------- #
# 1. Scrape controls
# --------------------------------------------------------------------------- #
st.subheader("1 · Scrape the live site" if LOCAL_MODE else "1 · Stored scrape")

cache = load_cache()
records = cache.get("records", {})
ok_records = {uid: r for uid, r in records.items() if r.get("ok")}

if not LOCAL_MODE:
    # Read-only (the deployed app): compare against the scrape committed to the repository, no live requests.
    if cache.get("scraped_at"):
        st.caption(
            f"Showing the scrape stored in the repository: **{len(records)}** pages, last scraped **{cache['scraped_at']}**. "
            "This app is read-only. To update it, re-scrape on a local machine with `ZOA_LOCAL_MODE=1` and push the result."
        )
    else:
        st.info("No scrape is stored in the repository yet.")
else:
    c1, c2 = st.columns([2, 3])
    with c1:
        if cache.get("scraped_at"):
            st.metric("Cached pages", len(records), help="Loaded from a previous scrape.")
            st.caption(f"Last scraped: **{cache['scraped_at']}**")
        else:
            st.info("No scrape cached yet. Fetch the sitemap and scrape to compare.")

    with c2:
        st.caption(
            "The site asks for ≤ 1 request/second, so a full crawl of ~300 pages "
            "takes about 5 minutes. Results are cached — you only do this once."
        )
        mode = st.radio(
            "What to scrape",
            ["Only missing / not-yet-cached", "Everything (re-scrape all)", "A small sample"],
            horizontal=False,
        )
        sample_n = 0
        if mode == "A small sample":
            sample_n = st.slider("Sample size", 5, 100, 20, step=5)
        full_rescrape = mode == "Everything (re-scrape all)"
        if full_rescrape:
            st.caption("A full re-scrape also visits every indicator of the CSV export that is not in the sitemap, so pages that were removed from the site are recorded as gone.")
        read_levels = st.checkbox("Also re-read the indicator levels from the sector pages", value=full_rescrape, disabled=mode == "A small sample")
        do_refresh = st.checkbox("Afterwards refresh the pipeline data so all other pages reflect the new state", value=full_rescrape, disabled=mode == "A small sample",
                                 help="Rebuilds the IndiKit metadata, rewrites the output files of the latest run, copies them to outputs/latest and re-exports Power BI. No LLM calls.")

        if st.button("🔎 Fetch sitemap & scrape", type="primary"):
            with st.spinner("Fetching sitemap…"):
                try:
                    id_url = fetch_sitemap_ids()
                except Exception as exc:  # noqa: BLE001 — surface any network error
                    st.error(f"Could not fetch sitemap: {exc}")
                    id_url = {}
            if id_url:
                st.session_state["sitemap_ids"] = list(id_url)
                cache = store_live_ids(id_url)  # persist for accurate coverage on reload
                if mode == "Only missing / not-yet-cached":
                    todo = {u: url for u, url in id_url.items() if u not in ok_records}
                elif mode == "A small sample":
                    todo = dict(list(id_url.items())[:sample_n])
                else:
                    todo = dict(id_url)
                    for uid in full["Unique ID"].astype(str):
                        if uid not in todo:
                            todo[uid] = records.get(uid, {}).get("url") or f"{BASE_URL}/indicator/{uid}"

                if not todo:
                    st.success("Everything in the sitemap is already cached.")
                else:
                    bar = st.progress(0.0, text=f"Scraping {len(todo)} pages…")

                    def _tick(done, total, uid):
                        bar.progress(done / total, text=f"Scraping {done}/{total} — id {uid}")

                    cache = crawl(todo, progress=_tick)
                    bar.empty()
                    records = cache.get("records", {})
                    ok_records = {uid: r for uid, r in records.items() if r.get("ok")}
                    gone = sum(1 for uid in todo if not records.get(uid, {}).get("ok"))
                    st.success(f"Scraped {len(todo)} pages: {len(todo) - gone} live, {gone} not found. Cache updated.")
                    if read_levels and mode != "A small sample":
                        lbar = st.progress(0.0, text="Reading indicator levels from the sector pages…")
                        try:
                            fetch_site_levels(progress=lambda d, t, m: lbar.progress(d / max(t, 1), text=m))
                            cache = load_cache()
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"Levels not refreshed: {exc}")
                        lbar.empty()
                    if do_refresh and mode != "A small sample":
                        refresh_pipeline_data()

    if cache.get("scraped_at"):
        rc1, rc2 = st.columns([3, 2])
        rc1.caption("Scraped elsewhere, or skipped the refresh above? Push the cached scrape through the pipeline so the other pages (Indicator Matching, Analysis IndiKit, Proposal Overview) and the Power BI export reflect it.")
        if rc2.button("🔄 Refresh pipeline data from the cached scrape"):
            refresh_pipeline_data()

# Live id list for coverage: freshly fetched sitemap (session) → cache → fallback.
sitemap_ids = st.session_state.get("sitemap_ids") or cache.get("live_ids")

if not records:
    st.stop()

# --------------------------------------------------------------------------- #
# 2. Coverage — what still exists on the site
# --------------------------------------------------------------------------- #
st.subheader("2 · Coverage")

csv_ids = set(full["Unique ID"].astype(str))
# The set of live ids: prefer the freshly fetched sitemap, else the scraped ok set.
live_ids = set(sitemap_ids) if sitemap_ids else set(ok_records)

removed = sorted(csv_ids - live_ids)      # in the CSV export, gone from the site
site_only = sorted(live_ids - csv_ids)    # on the site, missing from the CSV

m1, m2, m3, m4 = st.columns(4)
m1.metric("CSV unique indicators", len(csv_ids))
m2.metric("Live on site", len(live_ids))
m3.metric("In CSV, removed from site", len(removed))
m4.metric("On site, missing from CSV", len(site_only))

if removed:
    with st.expander(f"{len(removed)} indicators in the CSV but no longer on the site"):
        rem_df = full[full["Unique ID"].astype(str).isin(removed)][
            ["Unique ID", "Indicator Name", "Sector", "Indicator Level"]
        ].drop_duplicates()
        st.dataframe(rem_df, width="stretch", hide_index=True)

if site_only:
    with st.expander(f"{len(site_only)} indicators on the site but missing from the CSV"):
        so_df = pd.DataFrame(
            [{"Unique ID": u, "url": records.get(u, {}).get("url", "")} for u in site_only]
        )
        st.dataframe(so_df, width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# 3. Field-level drift
# --------------------------------------------------------------------------- #
st.subheader("3 · Field drift (CSV vs live page)")
st.caption(
    "Similarity is 0–1 (1 = identical after whitespace/case normalisation). "
    "Low French scores are expected where the CSV export has broken accents."
)

comp = build_comparison(df, ok_records)  # honours the active sidebar filter

if comp.empty:
    st.warning("No overlap between the current selection and the scraped pages yet.")
    st.stop()

sim_cols = ["name_sim", "wording_en_sim", "wording_fr_sim", "purpose_sim"]
a1, a2, a3, a4, a5 = st.columns(5)
a1.metric("Compared", len(comp))
a2.metric("Avg name", f"{comp['name_sim'].mean():.2f}")
a3.metric("Avg EN wording", f"{comp['wording_en_sim'].mean():.2f}")
a4.metric("Avg FR wording", f"{comp['wording_fr_sim'].mean():.2f}")
a5.metric("Avg purpose", f"{comp['purpose_sim'].mean():.2f}")

threshold = st.slider(
    "Flag rows below this similarity (on any compared field)", 0.0, 1.0, 0.90, 0.05
)
flagged = comp[comp[sim_cols].min(axis=1) < threshold].sort_values(
    sim_cols, ascending=True
)

st.caption(f"**{len(flagged)}** of {len(comp)} indicators drift below {threshold:.2f}.")
show_cols = [
    "unique_id", "name", "name_sim", "wording_en_sim",
    "wording_fr_sim", "purpose_sim", "section_match", "sector_match", "sub_sector_match",
]
st.dataframe(
    flagged[show_cols],
    width="stretch",
    hide_index=True,
    column_config={c: st.column_config.NumberColumn(format="%.2f") for c in sim_cols},
)

# --------------------------------------------------------------------------- #
# 4. Side-by-side inspector
# --------------------------------------------------------------------------- #
st.subheader("4 · Inspect one indicator")
name_by_id = dict(zip(comp["unique_id"], comp["name"]))
pick = st.selectbox(
    "Indicator",
    options=comp["unique_id"].tolist(),
    format_func=lambda u: f"{u} — {name_by_id.get(u, '')}",
)
row = comp.loc[comp["unique_id"] == pick].iloc[0]
csv_row = df[df["Unique ID"].astype(str) == str(pick)].iloc[0]
scraped = ok_records.get(str(pick), {})

st.caption(f"Live page: {scraped.get('url', '—')}")

FIELDS = [
    ("Indicator Name", "Indicator Name", "name", "name_sim"),
    ("Wording (EN)", "Indicator Wording - ENG", "wording_en", "wording_en_sim"),
    ("Wording (FR)", "Indicator Wording - FRA", "wording_fr", "wording_fr_sim"),
    ("Purpose", "What is its Purpose?", "purpose", "purpose_sim"),
]
for label, csv_col, scr_key, sim_key in FIELDS:
    st.markdown(f"**{label}** · similarity `{row[sim_key]:.2f}`")
    lft, rgt = st.columns(2)
    lft.caption("CSV export")
    lft.write(str(csv_row.get(csv_col, "")) or "—")
    rgt.caption("Live site")
    rgt.write(scraped.get(scr_key, "") or "—")
    st.divider()

# --------------------------------------------------------------------------- #
# 5. Indicator levels — can the site fill the CSV's gaps?
# --------------------------------------------------------------------------- #
st.subheader("5 · Indicator levels — can the site fill the gaps?")
st.caption(
    "Indicator *detail* pages don't state a level, but each sector listing can "
    "be filtered by level on the site — so we read the level(s) from there. This "
    "checks whether the site knows a level for the indicators the CSV is missing one."
)

levels = cache.get("levels", {})
if LOCAL_MODE:
    lc1, lc2 = st.columns([2, 3])
    with lc1:
        if cache.get("levels_at"):
            st.metric("Indicators with a site level", len(levels))
            st.caption(f"Levels read: **{cache['levels_at']}**")
        else:
            st.info("Levels not fetched yet.")
    with lc2:
        st.caption(
            "Reads all 9 sector pages × each level filter (~30 requests at 1 req/s, "
            "≈1 minute). Cached like the main scrape."
        )
        if st.button("🏷️ Fetch indicator levels from site"):
            bar = st.progress(0.0, text="Reading level filters…")

            def _ltick(done, total, msg):
                bar.progress(done / total, text=f"{done}/{total} — {msg}")

            levels = fetch_site_levels(progress=_ltick)
            bar.empty()
            cache = load_cache()
            st.success(f"Read levels for {len(levels)} indicators.")
elif cache.get("levels_at"):
    st.caption(f"Levels read from the site on **{cache['levels_at']}**: **{len(levels)}** indicators with a site level.")

level_comp = pd.DataFrame()
if not levels:
    st.info("Fetch levels above to see whether the site can fill the CSV's gaps." if LOCAL_MODE else "No indicator levels are stored in the repository's scrape.")
else:
    level_comp = build_level_comparison(df, levels)  # honours the active filter

    missing_mask = level_comp["csv_level"] == "(missing)"
    n_missing = int(missing_mask.sum())
    n_filled = int((level_comp["status"] == "site fills gap").sum())
    n_unknown = int((level_comp["status"] == "still unknown").sum())
    n_differ = int((level_comp["status"] == "differ").sum())

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Missing level in CSV", n_missing)
    g2.metric("Site can fill", n_filled)
    g3.metric("Unknown on both", n_unknown)
    g4.metric("CSV vs site differ", n_differ)

    st.markdown("**Indicators the CSV is missing a level for**")
    gaps = level_comp[missing_mask].sort_values(["status", "sector", "name"])
    if gaps.empty:
        st.success("No missing levels in the current selection.")
    else:
        st.dataframe(
            gaps[["unique_id", "name", "sector", "site_level", "status"]],
            width="stretch",
            hide_index=True,
        )
        if n_filled == 0:
            st.caption(
                "The site doesn't tag a level for any of these either — they're "
                "uncategorised on zoa.indikit.net as well, so it can't fill the gaps."
            )

    if n_differ:
        with st.expander(f"{n_differ} indicators where CSV and site levels disagree"):
            st.dataframe(
                level_comp[level_comp["status"] == "differ"][
                    ["unique_id", "name", "sector", "csv_level", "site_level"]
                ],
                width="stretch",
                hide_index=True,
            )

    with st.expander("Full level comparison (all statuses)"):
        st.dataframe(level_comp, width="stretch", hide_index=True)

# --------------------------------------------------------------------------- #
# 6. Persist + download
# --------------------------------------------------------------------------- #
coverage_summary = pd.DataFrame(
    [
        {"metric": "csv_unique", "count": len(csv_ids)},
        {"metric": "live_on_site", "count": len(live_ids)},
        {"metric": "removed_from_site", "count": len(removed)},
        {"metric": "site_only_missing_from_csv", "count": len(site_only)},
    ]
)

metrics = {
    "csv_unique": len(csv_ids),
    "live_on_site": len(live_ids),
    "removed_from_site": len(removed),
    "site_only": len(site_only),
    "compared": int(len(comp)),
    "flagged_below_threshold": int(len(flagged)),
    "avg_name_sim": round(float(comp["name_sim"].mean()), 3),
    "avg_wording_en_sim": round(float(comp["wording_en_sim"].mean()), 3),
    "avg_wording_fr_sim": round(float(comp["wording_fr_sim"].mean()), 3),
    "avg_purpose_sim": round(float(comp["purpose_sim"].mean()), 3),
}
tables = {"comparison": comp, "coverage_summary": coverage_summary}
if not level_comp.empty:
    metrics.update(
        {
            "levels_missing_in_csv": int((level_comp["csv_level"] == "(missing)").sum()),
            "levels_site_can_fill": int((level_comp["status"] == "site fills gap").sum()),
            "levels_unknown_on_both": int((level_comp["status"] == "still unknown").sum()),
            "levels_differ": int((level_comp["status"] == "differ").sum()),
        }
    )
    tables["level_comparison"] = level_comp

log_section(
    "Site Comparison",
    metrics=metrics,
    tables=tables,
    context={**active_filter_context(), "scraped_at": cache.get("scraped_at"),
             "levels_at": cache.get("levels_at"), "threshold": threshold},
)

d1, d2 = st.columns(2)
d1.download_button(
    "⬇ Download field comparison CSV",
    data=comp.to_csv(index=False).encode("utf-8"),
    file_name="site_comparison.csv",
    mime="text/csv",
)
if not level_comp.empty:
    d2.download_button(
        "⬇ Download level comparison CSV",
        data=level_comp.to_csv(index=False).encode("utf-8"),
        file_name="site_level_comparison.csv",
        mime="text/csv",
    )
