"""Page 2 — Coverage: where IndiKit is thin (the matcher's hard spots)."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis_log import log_section
from src.loader import split_levels
from src.plots import crosstab_heatmap
from src.state import active_filter_context, get_data, show_active_count

st.set_page_config(page_title="Coverage · IndiKit", layout="wide")
st.title("Coverage")
st.caption("Where is the taxonomy sparse? Those cells are where matching will struggle.")

full, df = get_data()
show_active_count(df, full)

if df.empty:
    st.warning("No indicators match the current filters.")
    st.stop()

# Explode the "Output | Outcome" mixed levels so each level is counted once.
exploded = split_levels(df)

# --- Sector x Indicator Level heatmap --------------------------------------
st.subheader("Sector × Indicator Level")
ct = pd.crosstab(exploded["Sector"], exploded["Indicator Level"])
st.plotly_chart(crosstab_heatmap(ct, ""), width="stretch")

# --- Coverage gap callouts -------------------------------------------------
st.subheader("Coverage gaps (zero indicators)")
gaps = ct.stack().reset_index(name="count")
gaps = gaps[gaps["count"] == 0][["Sector", "Indicator Level"]]
if gaps.empty:
    st.success("No empty (Sector × Level) cells in the current selection.")
else:
    st.caption("These (Sector, Level) combinations have no indicators — explicit gaps to cite.")
    st.dataframe(gaps, width="stretch", hide_index=True)

# --- Sector x Sub-Sector heatmap -------------------------------------------
st.subheader("Sector × Sub-Sector")
ct2 = pd.crosstab(df["Sector"], df["Sub-Sector"])
fig2 = crosstab_heatmap(ct2, "")
fig2.update_layout(height=500)
st.plotly_chart(fig2, width="stretch")

# --- Stacked level mix per sector ------------------------------------------
st.subheader("Indicator-level mix per sector")
mix = (
    exploded.groupby(["Sector", "Indicator Level"]).size().reset_index(name="count")
)
fig3 = px.bar(
    mix, x="Sector", y="count", color="Indicator Level", barmode="stack"
)
fig3.update_layout(xaxis_title="", yaxis_title="indicators")
st.plotly_chart(fig3, width="stretch")

st.download_button(
    "⬇ Download Sector × Level table",
    data=ct.to_csv().encode("utf-8"),
    file_name="coverage_sector_level.csv",
    mime="text/csv",
)

log_section(
    "Coverage",
    metrics={"empty_sector_level_cells": int(len(gaps))},
    tables={
        "sector_x_level": ct,
        "sector_x_sub_sector": ct2,
        "coverage_gaps": gaps,
        "level_mix_per_sector": mix,
    },
    indexed_tables=("sector_x_level", "sector_x_sub_sector"),
    context=active_filter_context(),
)
