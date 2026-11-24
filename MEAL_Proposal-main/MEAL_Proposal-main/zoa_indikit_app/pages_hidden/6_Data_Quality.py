"""Page 6 — Data Quality: make missing/inconsistent fields explicit."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis_log import log_section
from src.loader import load_indikit
from src.state import active_filter_context, get_data, show_active_count

st.set_page_config(page_title="Data Quality · IndiKit", layout="wide")
st.title("Data Quality")
st.caption("Quirks to either fix in the source or handle in the matcher.")

full, df = get_data()
show_active_count(df, full)

# Missing values are measured against the RAW load (before fillna in the loader).
raw = load_indikit()

# --- Missing values --------------------------------------------------------
st.subheader("Missing values per column")
st.caption("Categorical columns are filled with `(missing)` by the loader; this counts them too.")
missing = (df == "(missing)").sum() + df.isna().sum()
missing = missing[missing > 0].sort_values()
if missing.empty:
    st.success("No missing values in the current selection.")
else:
    miss_df = missing.rename_axis("column").reset_index(name="missing")
    st.plotly_chart(
        px.bar(miss_df, x="missing", y="column", orientation="h", text="missing"),
        width="stretch",
    )

# --- Indicator Level values (incl. mixed) ----------------------------------
st.subheader("Indicator Level values")
st.caption("Mixed values like `Output | Outcome` are split on Coverage/Semantic pages.")
lvl = df["Indicator Level"].value_counts(dropna=False).rename_axis("level").reset_index(name="count")
st.dataframe(lvl, width="stretch", hide_index=True)

# --- Language coverage -----------------------------------------------------
st.subheader("Language coverage")
en = df["Indicator Wording - ENG"].notna() & (df["Indicator Wording - ENG"].astype(str).str.strip() != "")
fr = df["Indicator Wording - FRA"].notna() & (df["Indicator Wording - FRA"].astype(str).str.strip() != "")
lang = pd.Series(
    {
        "EN + FR": int((en & fr).sum()),
        "EN only": int((en & ~fr).sum()),
        "FR only": int((~en & fr).sum()),
        "neither": int((~en & ~fr).sum()),
    }
).rename_axis("coverage").reset_index(name="count")
st.plotly_chart(px.bar(lang, x="coverage", y="count", text="count"), width="stretch")

# --- URL structural check --------------------------------------------------
st.subheader("Guidance URL check (structural)")
url_ok = df["Guidance"].astype(str).str.match(r"^https?://", na=False)
st.metric("Valid-looking URLs", f"{int(url_ok.sum())} / {len(df)}")
bad = df.loc[~url_ok, ["Unique ID", "Indicator Name", "Guidance"]]
if not bad.empty:
    st.caption("Rows whose Guidance is not a http(s) URL:")
    st.dataframe(bad, width="stretch", hide_index=True)

# --- Log the standing analysis ---------------------------------------------
missing_table = missing.rename_axis("column").reset_index(name="missing")
log_section(
    "Data Quality",
    metrics={
        "indicators": int(len(df)),
        "valid_guidance_urls": int(url_ok.sum()),
        "invalid_guidance_urls": int((~url_ok).sum()),
    },
    tables={
        "missing_values": missing_table,
        "indicator_level_values": lvl,
        "language_coverage": lang,
        "invalid_urls": bad,
    },
    context=active_filter_context(),
)

# --- Download cleaned CSV --------------------------------------------------
st.download_button(
    "⬇ Download cleaned CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="indikit_cleaned.csv",
    mime="text/csv",
)
