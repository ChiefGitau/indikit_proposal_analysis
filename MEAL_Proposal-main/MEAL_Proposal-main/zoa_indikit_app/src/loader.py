"""CSV loader + cleaning for the ZOA-IndiKit export.

`load_indikit()` is the single cached entry point used by every page.
`split_levels()` and `apply_filters()` are pure helpers (no Streamlit state)
so they can be unit-tested and reused freely.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Export indikit.csv"

CATEGORICAL_COLS = ["Section", "Sector", "Sub-Sector", "Indicator Level"]


@st.cache_data
def load_indikit(path=None) -> pd.DataFrame:
    """Read the IndiKit export, drop empty trailing columns, normalise types."""
    csv_path = Path(path) if path else DATA_PATH
    df = pd.read_csv(csv_path, encoding="latin-1")

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = [c.strip() for c in df.columns]

    df["Unique ID"] = df["Unique ID"].astype(str)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("(missing)").astype(str).str.strip()

    return df


def split_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Explode ``Output | Outcome`` style rows into one row per level.

    The original combined value is kept in ``Indicator Level (raw)`` for
    traceability; ``Indicator Level`` becomes a single atomic level.
    """
    out = df.copy()
    out["Indicator Level (raw)"] = out["Indicator Level"]
    out["Indicator Level"] = out["Indicator Level"].str.split(r"\s*\|\s*")
    out = out.explode("Indicator Level", ignore_index=True)
    out["Indicator Level"] = out["Indicator Level"].str.strip()
    return out


def apply_filters(df: pd.DataFrame, sectors=None, levels=None, sections=None) -> pd.DataFrame:
    """Return the subset of ``df`` matching any non-empty filter list.

    An empty / falsy list for a dimension means "no filter" on that dimension.
    """
    out = df
    if sections:
        out = out[out["Section"].isin(sections)]
    if sectors:
        out = out[out["Sector"].isin(sectors)]
    if levels:
        out = out[out["Indicator Level"].isin(levels)]
    return out
