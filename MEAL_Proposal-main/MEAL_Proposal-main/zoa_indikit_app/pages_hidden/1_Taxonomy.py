"""Page 1 — Taxonomy: the whole IndiKit structure on one screen."""

import plotly.express as px
import streamlit as st

from src.analysis_log import log_section
from src.plots import sector_bar
from src.state import active_filter_context, get_data, show_active_count

st.set_page_config(page_title="Taxonomy · IndiKit", layout="wide")
st.title("Taxonomy")

full, df = get_data()
show_active_count(df, full)

if df.empty:
    st.warning("No indicators match the current filters.")
    st.stop()

# --- KPI row ---------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Indicators", len(df))
k2.metric("Sectors", df["Sector"].nunique())
k3.metric("Sub-sectors", df["Sub-Sector"].nunique())
k4.metric("Sections", df["Section"].nunique())

# --- Hierarchy views -------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Sunburst")
    fig = px.sunburst(df, path=["Section", "Sector", "Sub-Sector"])
    fig.update_layout(margin=dict(t=10, l=0, r=0, b=0))
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Treemap")
    fig = px.treemap(df, path=["Section", "Sector", "Sub-Sector"])
    fig.update_layout(margin=dict(t=10, l=0, r=0, b=0))
    st.plotly_chart(fig, width="stretch")

# --- Indicators per sector (drill-down) ------------------------------------
st.subheader("Indicators per sector")
st.caption(
    "Note the class imbalance — Food Security & Livelihoods dwarfs sectors "
    "like Localisation. Click a bar to filter the table below."
)
bar_fig = sector_bar(df, title="")
event = st.plotly_chart(
    bar_fig, width="stretch", on_select="rerun", selection_mode="points"
)

selected_sectors = []
if event and event.get("selection", {}).get("points"):
    selected_sectors = [p["y"] for p in event["selection"]["points"]]

table = df[df["Sector"].isin(selected_sectors)] if selected_sectors else df
if selected_sectors:
    st.caption(f"Filtered to: {', '.join(selected_sectors)}")

st.dataframe(
    table[["Section", "Sector", "Sub-Sector", "Indicator Level", "Indicator Name"]],
    width="stretch",
    hide_index=True,
)

sector_counts = (
    df["Sector"].value_counts().rename_axis("Sector").reset_index(name="count")
)
subsector_counts = (
    df["Sub-Sector"].value_counts().rename_axis("Sub-Sector").reset_index(name="count")
)
log_section(
    "Taxonomy",
    metrics={
        "indicators": len(df),
        "sectors": int(df["Sector"].nunique()),
        "sub_sectors": int(df["Sub-Sector"].nunique()),
        "sections": int(df["Section"].nunique()),
    },
    tables={
        "indicators_per_sector": sector_counts,
        "indicators_per_sub_sector": subsector_counts,
    },
    context=active_filter_context(),
)
