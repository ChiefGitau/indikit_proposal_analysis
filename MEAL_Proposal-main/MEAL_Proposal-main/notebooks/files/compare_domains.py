"""
Compare two (or more) scraped IndiKit domains.

Usage:
  python compare_domains.py data/tdh_indikit_net/indicators.json data/zoa_indikit_net/indicators.json

Produces:
  comparison/overview.csv          — sector coverage per domain
  comparison/shared_indicators.csv — indicators appearing in multiple domains (fuzzy title match)
  comparison/unique_<domain>.csv   — indicators found only in one domain
  comparison/field_coverage.csv    — which metadata fields each domain fills in
"""

import json
import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd


def normalize(title: str) -> str:
    """Normalize titles so 'Births assisted... (%)' matches 'Births assisted... (#)' variants loosely."""
    t = title.lower()
    t = re.sub(r"[\(\)#%]", " ", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def token_similarity(a: str, b: str) -> float:
    """Jaccard similarity on word tokens — cheap fuzzy match, no dependencies."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def main(paths):
    out = Path("comparison")
    out.mkdir(exist_ok=True)

    datasets = {}
    for p in paths:
        records = json.loads(Path(p).read_text(encoding="utf-8"))
        df = pd.DataFrame(records)
        domain = df["domain"].iloc[0]
        df["norm_title"] = df["title"].fillna("").map(normalize)
        datasets[domain] = df

    # 1. Sector coverage overview
    rows = []
    for domain, df in datasets.items():
        for sector, grp in df.groupby("sector", dropna=False):
            rows.append({"domain": domain, "sector": sector, "n_indicators": len(grp)})
    overview = pd.DataFrame(rows).pivot_table(
        index="sector", columns="domain", values="n_indicators", fill_value=0
    )
    overview.to_csv(out / "overview.csv")
    print("Sector coverage:\n", overview, "\n")

    # 2. Shared vs unique indicators (pairwise fuzzy match on titles)
    shared_rows = []
    matched = {d: set() for d in datasets}
    for (da, dfa), (db, dfb) in combinations(datasets.items(), 2):
        for ia, ra in dfa.iterrows():
            best, best_score = None, 0.0
            for ib, rb in dfb.iterrows():
                s = token_similarity(ra["norm_title"], rb["norm_title"])
                if s > best_score:
                    best, best_score = rb, s
            if best is not None and best_score >= 0.6:
                shared_rows.append({
                    "domain_a": da, "title_a": ra["title"], "sector_a": ra.get("sector"),
                    "domain_b": db, "title_b": best["title"], "sector_b": best.get("sector"),
                    "similarity": round(best_score, 2),
                })
                matched[da].add(ra["title"])
                matched[db].add(best["title"])
    pd.DataFrame(shared_rows).to_csv(out / "shared_indicators.csv", index=False)
    print(f"Shared indicators (fuzzy ≥0.6): {len(shared_rows)}")

    # 3. Unique per domain
    for domain, df in datasets.items():
        unique = df[~df["title"].isin(matched[domain])]
        fname = out / f"unique_{domain.replace('.', '_')}.csv"
        unique[["title", "sector", "url"]].to_csv(fname, index=False)
        print(f"Unique to {domain}: {len(unique)}")

    # 4. Field/metadata coverage — reveals each domain's "focus"
    rows = []
    for domain, df in datasets.items():
        for col in df.columns:
            if col in ("norm_title",):
                continue
            filled = df[col].notna().sum() if col in df else 0
            rows.append({"domain": domain, "field": col, "filled": filled, "total": len(df),
                         "pct": round(100 * filled / len(df), 1)})
    pd.DataFrame(rows).to_csv(out / "field_coverage.csv", index=False)
    print(f"\nAll outputs written to {out}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
