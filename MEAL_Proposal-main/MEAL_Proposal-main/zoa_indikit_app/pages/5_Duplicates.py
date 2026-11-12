"""Page 5 — Duplicates & near-duplicates: ambiguous entries for the matcher."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis_log import log_section
from src.embeddings import embed_indicators
from src.state import active_filter_context, get_data, show_active_count

st.set_page_config(page_title="Duplicates · IndiKit", layout="wide")
st.title("Duplicates & Near-Duplicates")
st.caption("IndiKit entries that look almost identical — the matcher will see these as ambiguous.")

full, df = get_data()
show_active_count(df, full)

if len(df) < 2:
    st.warning("Need at least 2 indicators to compare.")
    st.stop()

WORDING = "Indicator Wording - ENG"
embed_text = df[WORDING].fillna(df["Indicator Name"]).astype(str)

if not st.checkbox("Compute similarities", value=True):
    st.stop()

vecs, emb_method = embed_indicators(tuple(embed_text.tolist()))
st.caption(f"Embeddings: **{emb_method}**")
if "TF-IDF" in emb_method:
    st.warning(
        "Running on the lexical (TF-IDF) fallback — near-duplicates are matched "
        "on shared words, not meaning. Install `sentence-transformers` for "
        "semantic similarity (see the README)."
    )
sim = vecs @ vecs.T  # normalised vectors → cosine similarity

# Off-diagonal upper triangle only.
iu, ju = np.triu_indices(len(sim), k=1)
sims = sim[iu, ju]

# --- Distribution ----------------------------------------------------------
st.subheader("Distribution of pairwise similarity")
st.plotly_chart(
    px.histogram(pd.DataFrame({"cosine similarity": sims}), x="cosine similarity", nbins=50),
    width="stretch",
)

# --- Pairs above threshold -------------------------------------------------
threshold = st.slider("Similarity threshold", 0.50, 1.0, 0.90, 0.01)
mask = sims >= threshold
names = df["Indicator Name"].values
ids = df["Unique ID"].values
sectors = df["Sector"].values

pairs = pd.DataFrame(
    {
        "similarity": np.round(sims[mask], 3),
        "Indicator A": names[iu[mask]],
        "ID A": ids[iu[mask]],
        "Sector A": sectors[iu[mask]],
        "Indicator B": names[ju[mask]],
        "ID B": ids[ju[mask]],
        "Sector B": sectors[ju[mask]],
    }
).sort_values("similarity", ascending=False, ignore_index=True)

st.subheader(f"{len(pairs)} pair(s) at ≥ {threshold:.2f}")
st.dataframe(pairs, width="stretch", hide_index=True)

if not pairs.empty:
    st.download_button(
        "⬇ Download duplicate pairs CSV",
        data=pairs.to_csv(index=False).encode("utf-8"),
        file_name=f"duplicate_pairs_ge_{threshold:.2f}.csv",
        mime="text/csv",
    )

sim_summary = pd.Series(sims).describe().rename_axis("stat").reset_index(name="value")
log_section(
    "Duplicates",
    metrics={
        "embedding_method": emb_method,
        "threshold": float(threshold),
        "pairs_above_threshold": int(len(pairs)),
        "max_similarity": round(float(sims.max()), 4),
    },
    tables={
        "duplicate_pairs": pairs,
        "similarity_summary": sim_summary,
    },
    context=active_filter_context(),
)
