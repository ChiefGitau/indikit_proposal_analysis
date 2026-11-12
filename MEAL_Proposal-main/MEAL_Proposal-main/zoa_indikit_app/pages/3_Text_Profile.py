"""Page 3 — Text Profile: wording patterns that drive the rule-based extractor."""

import re

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

from src.analysis_log import log_section
from src.state import active_filter_context, get_data, show_active_count

st.set_page_config(page_title="Text Profile · IndiKit", layout="wide")
st.title("Text Profile")
st.caption("Wording patterns that feed the rule-based extractor (regex starters, n-grams).")

full, df = get_data()
show_active_count(df, full)

WORDING = "Indicator Wording - ENG"
text = df[WORDING].dropna().astype(str)

if text.empty:
    st.warning("No English wording available for the current filters.")
    st.stop()

# --- Length distribution ---------------------------------------------------
st.subheader("Wording length")
lengths = pd.DataFrame(
    {
        "words": text.str.split().apply(len),
        "characters": text.str.len(),
        "Sector": df.loc[text.index, "Sector"].values,
    }
)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        px.histogram(lengths, x="words", nbins=30, title="Words per indicator"),
        width="stretch",
    )
with c2:
    st.plotly_chart(
        px.histogram(lengths, x="characters", nbins=30, title="Characters per indicator"),
        width="stretch",
    )

# --- Quantifier coverage ---------------------------------------------------
st.subheader("Quantifier patterns")
st.caption("Indicator-starter phrases — candidates for `src/extract/rules.py`.")
patterns = {
    "number of": r"^\s*number of\b",
    "% of / percentage": r"^\s*(?:%|percentage)",
    "proportion of": r"^\s*proportion of\b",
    "rate of": r"^\s*rate of\b",
    "average / mean": r"^\s*(?:average|mean)\b",
    "amount of": r"^\s*amount of\b",
    "frequency of": r"^\s*frequency of\b",
    "ratio of": r"^\s*ratio of\b",
}
lower = text.str.lower()
counts = {
    label: int(lower.str.contains(pat, regex=True).sum()) for label, pat in patterns.items()
}
qdf = pd.DataFrame({"pattern": list(counts), "count": list(counts.values())})
matched = int(sum(counts.values()))
st.plotly_chart(
    px.bar(qdf.sort_values("count"), x="count", y="pattern", orientation="h", text="count"),
    width="stretch",
)
st.caption(f"{matched} of {len(text)} indicators start with a recognised quantifier phrase.")

# --- Top n-grams -----------------------------------------------------------
st.subheader("Top n-grams")
ngram_choice = st.radio(
    "n-gram range", ["unigrams", "bigrams", "trigrams"], horizontal=True
)
ngram_range = {"unigrams": (1, 1), "bigrams": (2, 2), "trigrams": (3, 3)}[ngram_choice]
terms = None
try:
    vec = CountVectorizer(
        ngram_range=ngram_range, stop_words="english", min_df=min(3, len(text))
    )
    mat = vec.fit_transform(text)
    freqs = mat.sum(axis=0).A1
    terms = pd.DataFrame({"term": vec.get_feature_names_out(), "count": freqs})
    terms = terms.sort_values("count", ascending=False).head(25)
    st.plotly_chart(
        px.bar(terms.sort_values("count"), x="count", y="term", orientation="h"),
        width="stretch",
    )
except ValueError:
    st.info("Not enough text to extract n-grams for the current filters.")

# --- Log the standing analysis (search results are excluded — they're ad hoc).
length_summary = lengths[["words", "characters"]].describe().rename_axis("stat").reset_index()
log_tables = {
    "length_summary": length_summary,
    "quantifier_patterns": qdf,
}
if terms is not None:
    log_tables[f"top_{ngram_choice}"] = terms
log_section(
    "Text Profile",
    metrics={
        "indicators_with_english": int(len(text)),
        "quantifier_matches": matched,
        "median_words": float(lengths["words"].median()),
    },
    tables=log_tables,
    context={**active_filter_context(), "ngram_view": ngram_choice},
)

# --- Optional word cloud ---------------------------------------------------
with st.expander("Word cloud"):
    try:
        from wordcloud import WordCloud

        wc = WordCloud(width=900, height=350, background_color="white").generate(
            " ".join(text.tolist())
        )
        st.image(wc.to_array(), width="stretch")
    except ImportError:
        st.info("Install `wordcloud` to enable this view.")

# --- Search ----------------------------------------------------------------
st.subheader("Search wording")
query = st.text_input("Substring search over English wording")
if query:
    hits = df[df[WORDING].astype(str).str.contains(re.escape(query), case=False, na=False)]
    st.caption(f"{len(hits)} match(es).")
    st.dataframe(
        hits[["Sector", "Indicator Name", WORDING, "Guidance"]],
        width="stretch",
        hide_index=True,
        column_config={"Guidance": st.column_config.LinkColumn("Guidance")},
    )
