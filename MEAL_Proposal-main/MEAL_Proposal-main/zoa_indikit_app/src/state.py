"""Shared page helpers: pull the dataframe + active filters from session_state.

Every page calls :func:`get_data` so the global sidebar filters set on the
landing page propagate everywhere, and so a page opened directly (without
visiting the landing page first) still works.
"""

import streamlit as st

from src.loader import apply_filters, load_indikit


def get_data():
    """Return ``(full_df, filtered_df)`` honouring the global sidebar filters."""
    df = st.session_state.get("df")
    if df is None:
        df = load_indikit()
        st.session_state["df"] = df

    filters = st.session_state.get("filters", {})
    filtered = apply_filters(
        df,
        sectors=filters.get("sectors"),
        levels=filters.get("levels"),
        sections=filters.get("sections"),
    )
    return df, filtered


def active_filter_context():
    """Return the active sidebar filters as a JSON-friendly dict for the log."""
    f = st.session_state.get("filters", {})
    return {
        "sections": f.get("sections") or "all",
        "sectors": f.get("sectors") or "all",
        "levels": f.get("levels") or "all",
    }


def show_active_count(filtered_df, full_df):
    """Render a caption showing how many indicators the filters leave active."""
    n, total = len(filtered_df), len(full_df)
    if n == total:
        st.caption(f"Showing all **{total}** indicators (no filters active).")
    else:
        st.caption(f"Showing **{n}** of **{total}** indicators (filters active).")
