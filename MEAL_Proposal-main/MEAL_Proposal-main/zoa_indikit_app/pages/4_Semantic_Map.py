"""Page 4 — Semantic Map: UMAP of indicator embeddings, coloured by sector."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis_log import log_section
from src.embeddings import embed_indicators, project_2d
from src.state import active_filter_context, get_data, show_active_count

st.set_page_config(page_title="Semantic Map · IndiKit", layout="wide")
st.title("Semantic Map")
st.caption("2D UMAP of indicator embeddings. The single most informative chart.")

full, df = get_data()
show_active_count(df, full)

if len(df) < 5:
    st.warning("Need at least 5 indicators to build a meaningful projection.")
    st.stop()

WORDING = "Indicator Wording - ENG"
# Fall back to the indicator name where English wording is missing.
embed_text = df[WORDING].fillna(df["Indicator Name"]).astype(str)

color_by = st.radio(
    "Colour by", ["Sector", "Indicator Level", "Sub-Sector"], horizontal=True
)

if not st.checkbox("Build / refresh the map", value=True):
    st.stop()

vecs, emb_method = embed_indicators(tuple(embed_text.tolist()))
xy, proj_method = project_2d(vecs)
st.caption(f"Embeddings: **{emb_method}** · Projection: **{proj_method}**")
if "TF-IDF" in emb_method or "PCA" in proj_method:
    st.warning(
        "Running on the lexical fallback. For semantic embeddings install "
        "`sentence-transformers` (and `umap-learn` for UMAP) — see the README."
    )

plot_df = pd.DataFrame(
    {
        "x": xy[:, 0],
        "y": xy[:, 1],
        "Sector": df["Sector"].values,
        "Indicator Level": df["Indicator Level"].values,
        "Sub-Sector": df["Sub-Sector"].values,
        "Indicator Name": df["Indicator Name"].values,
        "Wording": embed_text.str.slice(0, 120).values,
    }
)

fig = px.scatter(
    plot_df,
    x="x",
    y="y",
    color=color_by,
    hover_data={"Indicator Name": True, "Wording": True, "x": False, "y": False},
    height=650,
)
fig.update_traces(marker=dict(size=8, opacity=0.75))
fig.update_layout(xaxis_title="", yaxis_title="", legend_title=color_by)
st.plotly_chart(fig, width="stretch")

log_section(
    "Semantic Map",
    metrics={
        "indicators": int(len(plot_df)),
        "embedding_method": emb_method,
        "projection_method": proj_method,
    },
    tables={
        "coordinates": plot_df[
            ["Indicator Name", "Sector", "Indicator Level", "Sub-Sector", "x", "y"]
        ]
    },
    context={**active_filter_context(), "coloured_by": color_by},
)

st.info(
    "**How to read this:** clean, separated clusters per sector → semantic "
    "matching alone will work well. Overlapping clouds → the matcher needs "
    "sector-conditioning (filter candidates by sector before ranking). This "
    "informs the matcher design decision in Phase 5 of the coding plan."
)
