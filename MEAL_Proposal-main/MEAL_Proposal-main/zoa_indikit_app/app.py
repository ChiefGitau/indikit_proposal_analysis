"""ZOA-IndiKit Indicator Profiler — Streamlit entry point.

Landing page. Loads the IndiKit export, exposes the global sidebar filters
(Section / Sector / Indicator Level) that propagate to every sub-page via
``st.session_state``, and offers a download of the cleaned CSV.
"""

import io
import zipfile

import pandas as pd
import streamlit as st

from src.analysis_log import log_section
from src.loader import apply_filters, load_indikit, split_levels

st.set_page_config(page_title="ZOA-IndiKit Profiler", layout="wide")
st.title("ZOA-IndiKit Indicator Profiler")

df = load_indikit()

with st.sidebar:
    st.header("Filters")
    st.caption("Applied across every page.")
    sections = st.multiselect("Section", sorted(df["Section"].unique()))
    sectors = st.multiselect("Sector", sorted(df["Sector"].unique()))
    levels = st.multiselect("Indicator Level", sorted(df["Indicator Level"].unique()))

# Stash for the sub-pages.
st.session_state["filters"] = dict(sections=sections, sectors=sectors, levels=levels)
st.session_state["df"] = df

filtered = apply_filters(df, sectors=sectors, levels=levels, sections=sections)

c1, c2, c3 = st.columns(3)
c1.metric("Indicators loaded", len(df))
c2.metric("Active after filters", len(filtered))
c3.metric("Sectors", df["Sector"].nunique())

st.caption(
    "Use the pages in the sidebar to explore the taxonomy, coverage, text "
    "profile, semantic map, near-duplicates, and data quality. Filters on the "
    "left apply everywhere."
)

st.dataframe(filtered, width="stretch", hide_index=True)

log_section(
    "Overview",
    metrics={
        "indicators_loaded": len(df),
        "active_after_filters": len(filtered),
        "sectors": int(df["Sector"].nunique()),
    },
    tables={"filtered_indicators": filtered},
    context={
        "sections": sections or "all",
        "sectors": sectors or "all",
        "levels": levels or "all",
    },
)

col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "⬇ Download cleaned CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="indikit_cleaned.csv",
        mime="text/csv",
    )


def _build_report_zip(data: pd.DataFrame) -> bytes:
    """Bundle the cheap-to-compute findings into a single zip (no ML needed)."""
    exploded = split_levels(data)
    coverage = pd.crosstab(exploded["Sector"], exploded["Indicator Level"])
    sector_counts = data["Sector"].value_counts().rename_axis("Sector").reset_index(name="count")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("indikit_cleaned.csv", data.to_csv(index=False))
        zf.writestr("coverage_sector_level.csv", coverage.to_csv())
        zf.writestr("indicators_per_sector.csv", sector_counts.to_csv(index=False))
        zf.writestr(
            "README.txt",
            "ZOA-IndiKit profiler export\n"
            f"Indicators: {len(data)}\n"
            "Files: cleaned CSV, Sector x Level coverage, per-sector counts.\n"
            "Embeddings / duplicate pairs export from the respective pages.\n",
        )
    return buf.getvalue()


with col_b:
    st.download_button(
        "⬇ Download report (zip)",
        data=_build_report_zip(filtered),
        file_name="indikit_report.zip",
        mime="application/zip",
    )


## save the children: https://indicators.savethechildren.net/
## winrock : https://winrock.indikit.net/

## welt hunger hlife: https://whh.indikit.net/

## helvatas: https://helvetas.indikit.net/

## tearfunds: https://tearfund.indikit.net/
## adra : https://adra.indikit.net/

## islamic relief world wide : https://irw.indikit.net/

## Terre des hommes : https://tdh.indikit.net/

## find the conntact information on portals and insight on indicators and how they use it