"""Headless regeneration of the analysis log.

Reproduces what each Streamlit analysis page computes and writes it to the
shared log at data/indicators/data_analysis/indikit_analysis_log/, without
needing to click through the UI. Run after changing the data or the embedding
stack:

    cd zoa_indikit_app
    .venv/bin/python scripts/regenerate_log.py

Uses the same loader / embeddings / log modules as the app, so results match
the pages (real sentence-transformer embeddings + UMAP when installed).
"""

import sys
from pathlib import Path

import pandas as pd

# Make `src` importable when run from the app root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis_log import log_section  # noqa: E402
from src.embeddings import embed_indicators, project_2d  # noqa: E402
from src.loader import load_indikit, split_levels  # noqa: E402

WORDING = "Indicator Wording - ENG"
ALL = {"sections": "all", "sectors": "all", "levels": "all"}


def main():
    df = load_indikit()
    print(f"Loaded {len(df)} indicators")

    # --- Taxonomy ----------------------------------------------------------
    log_section(
        "Taxonomy",
        metrics={
            "indicators": len(df),
            "sectors": int(df["Sector"].nunique()),
            "sub_sectors": int(df["Sub-Sector"].nunique()),
            "sections": int(df["Section"].nunique()),
        },
        tables={
            "indicators_per_sector": df["Sector"].value_counts().rename_axis("Sector").reset_index(name="count"),
            "indicators_per_sub_sector": df["Sub-Sector"].value_counts().rename_axis("Sub-Sector").reset_index(name="count"),
        },
        context=ALL,
    )
    print("✓ Taxonomy")

    # --- Coverage ----------------------------------------------------------
    exploded = split_levels(df)
    ct = pd.crosstab(exploded["Sector"], exploded["Indicator Level"])
    ct2 = pd.crosstab(df["Sector"], df["Sub-Sector"])
    gaps = ct.stack().reset_index(name="count")
    gaps = gaps[gaps["count"] == 0][["Sector", "Indicator Level"]]
    mix = exploded.groupby(["Sector", "Indicator Level"]).size().reset_index(name="count")
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
        context=ALL,
    )
    print("✓ Coverage")

    # --- Text Profile ------------------------------------------------------
    text = df[WORDING].dropna().astype(str)
    lengths = pd.DataFrame({"words": text.str.split().apply(len), "characters": text.str.len()})
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
    counts = {label: int(lower.str.contains(pat, regex=True).sum()) for label, pat in patterns.items()}
    qdf = pd.DataFrame({"pattern": list(counts), "count": list(counts.values())})
    matched = int(sum(counts.values()))

    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(ngram_range=(1, 1), stop_words="english", min_df=min(3, len(text)))
    mat = vec.fit_transform(text)
    terms = pd.DataFrame({"term": vec.get_feature_names_out(), "count": mat.sum(axis=0).A1})
    terms = terms.sort_values("count", ascending=False).head(25)
    length_summary = lengths[["words", "characters"]].describe().rename_axis("stat").reset_index()
    log_section(
        "Text Profile",
        metrics={
            "indicators_with_english": int(len(text)),
            "quantifier_matches": matched,
            "median_words": float(lengths["words"].median()),
        },
        tables={"length_summary": length_summary, "quantifier_patterns": qdf, "top_unigrams": terms},
        context={**ALL, "ngram_view": "unigrams"},
    )
    print("✓ Text Profile")

    # --- Embeddings (shared by Semantic Map + Duplicates) ------------------
    embed_text = df[WORDING].fillna(df["Indicator Name"]).astype(str)
    vecs, emb_method = embed_indicators(tuple(embed_text.tolist()))
    print(f"  embeddings: {emb_method}  shape={vecs.shape}")

    # --- Semantic Map ------------------------------------------------------
    xy, proj_method = project_2d(vecs)
    print(f"  projection: {proj_method}")
    coords = pd.DataFrame(
        {
            "Indicator Name": df["Indicator Name"].values,
            "Sector": df["Sector"].values,
            "Indicator Level": df["Indicator Level"].values,
            "Sub-Sector": df["Sub-Sector"].values,
            "x": xy[:, 0],
            "y": xy[:, 1],
        }
    )
    log_section(
        "Semantic Map",
        metrics={"indicators": int(len(coords)), "embedding_method": emb_method, "projection_method": proj_method},
        tables={"coordinates": coords},
        context={**ALL, "coloured_by": "Sector"},
    )
    print("✓ Semantic Map")

    # --- Duplicates --------------------------------------------------------
    import numpy as np

    sim = vecs @ vecs.T
    iu, ju = np.triu_indices(len(sim), k=1)
    sims = sim[iu, ju]
    threshold = 0.90
    mask = sims >= threshold
    names, ids, sectors = df["Indicator Name"].values, df["Unique ID"].values, df["Sector"].values
    pairs = pd.DataFrame(
        {
            "similarity": np.round(sims[mask], 3),
            "Indicator A": names[iu[mask]], "ID A": ids[iu[mask]], "Sector A": sectors[iu[mask]],
            "Indicator B": names[ju[mask]], "ID B": ids[ju[mask]], "Sector B": sectors[ju[mask]],
        }
    ).sort_values("similarity", ascending=False, ignore_index=True)
    sim_summary = pd.Series(sims).describe().rename_axis("stat").reset_index(name="value")
    log_section(
        "Duplicates",
        metrics={
            "embedding_method": emb_method,
            "threshold": float(threshold),
            "pairs_above_threshold": int(len(pairs)),
            "max_similarity": round(float(sims.max()), 4),
        },
        tables={"duplicate_pairs": pairs, "similarity_summary": sim_summary},
        context=ALL,
    )
    print(f"✓ Duplicates ({len(pairs)} pairs ≥ {threshold})")

    # --- Data Quality ------------------------------------------------------
    missing = (df == "(missing)").sum() + df.isna().sum()
    missing = missing[missing > 0].sort_values()
    missing_table = missing.rename_axis("column").reset_index(name="missing")
    lvl = df["Indicator Level"].value_counts(dropna=False).rename_axis("level").reset_index(name="count")
    en = df[WORDING].notna() & (df[WORDING].astype(str).str.strip() != "")
    fr = df["Indicator Wording - FRA"].notna() & (df["Indicator Wording - FRA"].astype(str).str.strip() != "")
    lang = pd.Series(
        {"EN + FR": int((en & fr).sum()), "EN only": int((en & ~fr).sum()),
         "FR only": int((~en & fr).sum()), "neither": int((~en & ~fr).sum())}
    ).rename_axis("coverage").reset_index(name="count")
    url_ok = df["Guidance"].astype(str).str.match(r"^https?://", na=False)
    bad = df.loc[~url_ok, ["Unique ID", "Indicator Name", "Guidance"]]
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
        context=ALL,
    )
    print("✓ Data Quality")
    print("\nLog regenerated.")


if __name__ == "__main__":
    main()
