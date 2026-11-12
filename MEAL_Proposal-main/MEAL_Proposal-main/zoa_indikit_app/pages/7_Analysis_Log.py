"""Page 7 — Analysis Log: every section's collected data, for exploration + sharing.

Each analysis page writes its tables + metrics to a durable on-disk log. This
page reads the whole log back so the results can be browsed and exported as a
single zip, regardless of which pages were opened this session.
"""

import streamlit as st

from src.analysis_log import clear_log, export_zip, read_log, read_table

st.set_page_config(page_title="Analysis Log · IndiKit", layout="wide")
st.title("Analysis Log")
st.caption(
    "Everything each section computed, collected for later exploration and "
    "sharing. Visit a page to (re)populate its entry with the active filters."
)

log = read_log()

if not log:
    st.info(
        "The log is empty. Open the Taxonomy, Coverage, Text Profile, Semantic "
        "Map, Duplicates and Data Quality pages — each records its results here."
    )
    st.stop()

# --- Top actions -----------------------------------------------------------
c1, c2, c3 = st.columns([2, 2, 6])
with c1:
    st.download_button(
        "⬇ Download full log (zip)",
        data=export_zip(),
        file_name="indikit_analysis_log.zip",
        mime="application/zip",
    )
with c2:
    if st.button("🗑 Clear log"):
        clear_log()
        st.rerun()
with c3:
    st.metric("Sections logged", len(log))

# --- Per-section browser ---------------------------------------------------
for section, entry in log.items():
    context = entry.get("context", {})
    updated = context.get("updated", "—")
    with st.expander(f"📊 {section}  ·  updated {updated}", expanded=False):
        ctx = {k: v for k, v in context.items() if k != "updated"}
        if ctx:
            st.caption("Context: " + ", ".join(f"**{k}** = {v}" for k, v in ctx.items()))

        metrics = list(entry.get("metrics", {}).items())
        for i in range(0, len(metrics), 4):
            row = metrics[i : i + 4]
            cols = st.columns(len(row))
            for col, (key, val) in zip(cols, row):
                col.metric(key.replace("_", " "), val)

        for name, info in entry.get("tables", {}).items():
            st.markdown(f"**{name}** · {info['rows']} rows")
            try:
                tbl = read_table(info["path"])
                st.dataframe(tbl, width="stretch", hide_index=True)
                st.download_button(
                    f"⬇ {name}.csv",
                    data=tbl.to_csv(index=False).encode("utf-8"),
                    file_name=f"{section}_{name}.csv".replace(" ", "_"),
                    mime="text/csv",
                    key=f"dl_{section}_{name}",
                )
            except FileNotFoundError:
                st.warning(f"Table file missing: {info['path']}")
