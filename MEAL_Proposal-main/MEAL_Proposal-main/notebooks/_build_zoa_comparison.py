"""One-shot generator for zoa_comparison.ipynb — deleted after use."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip()))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip()))


md('''
# ZOA vs all other IndiKit sites

`base_comparison.ipynb` measured every site against the base library (www.indikit.net). This notebook flips the reference point: **ZOA's library (303 indicators) is the anchor**, compared against the other 9 sites from `files/links` — the base plus adra, irw, tearfund, whh, winrock, tdh, helvetas and savethechildren. It answers:

1. which ZOA indicators other organizations also use — identically, reworded, or not at all;
2. how *standard* each ZOA indicator is (adoption count across sites) and which are ZOA-only;
3. how ZOA's wording drifts from sites that share its indicators;
4. what the others measure that ZOA doesn't — candidate gaps.

Method and plan: see `zoa_comparison_plan.md`. Uses the scraped snapshots in `data/` (no scraping); tables are written to `comparison_zoa/`.

**Caveat:** helvetas, savethechildren and tdh run custom platforms — they can only be matched via embeddings, so their "shared" counts mean *topical similarity*, not confirmed shared indicators.
''')

code('''
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ZOA = "zoa_indikit_net"
BASE = "www_indikit_net"

frames = {p.parent.name: pd.read_json(p) for p in sorted(Path("data").glob("*/indicators.json"))}
OTHERS = sorted((d for d in frames if d != ZOA), key=lambda d: -len(frames[d]))
zoa_df = frames[ZOA]

OUT = Path("comparison_zoa")
OUT.mkdir(exist_ok=True)


def short(domain):
    return domain.replace("indicators_", "").replace("_indikit_net", "").replace("_net", "")


# chart style (dataviz reference palette, light mode)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
C1, C2, C3 = "#2a78d6", "#1baf7a", "#eda100"  # categorical slots 1-3: blue, aqua, yellow
SEQ_BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.titlesize": 12,
    "legend.frameon": False, "font.family": "sans-serif", "font.size": 10,
})

print(f"zoa: {len(zoa_df)} indicators | compared against: {[short(d) for d in OTHERS]}")
''')

code('''
# Per-site overview. Sites on the IndiKit engine share the "How to Collect..."
# section verbatim; the others are custom platforms.
CLONE_MARKER = "section:How to Collect and Analyse the Required Data"

overview = pd.DataFrame(
    {
        "indicators": len(frames[d]),
        "sectors": frames[d]["sector"].nunique(),
        "template": "IndiKit clone" if CLONE_MARKER in frames[d].columns else "custom",
    }
    for d in [ZOA] + OTHERS
).set_axis([short(d) for d in [ZOA] + OTHERS])
overview
''')

md('''
## Matching ZOA's indicators to each site

Same three-tier ladder as `base_comparison.ipynb`, run pairwise ZOA → site: ① exact slug, ② slug text (numeric ID stripped) or exact normalized title, ③ embedding cosine ≥ 0.80 over the meaning-bearing fields. Below the cutoff = the indicator is absent from that site. Multilingual wording sections are reduced to their English text first.
''')

code('''
# Normalization helpers (identical to base_comparison.ipynb)
SEMANTIC_PATTERNS = ("wording", "phrasing", "purpose", "definition", "full statement")
WORDING_PATTERNS = ("wording", "phrasing", "full statement")
LANG_LABELS = {"english", "french", "arabic", "spanish", "portuguese", "français", "francais"}


def text_slug(slug):
    return re.sub(r"^\\d+-", "", str(slug))


def norm(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\\s+", " ", s).strip()


def english_only(text):
    # Sites publish sections multilingually in several layouts:
    #   block:       "English\\nFrench\\n<en>\\n<fr>"            (zoa, irw, adra)
    #   interleaved: "English:\\n<en>\\nFrench:\\n<fr>"          (whh)
    #   tab residue: "INDICATOR PHRASING\\nFrançais\\n<en>\\n…"  (tearfund)
    # Reduce every layout to the English text.
    lines = [l.strip() for l in text.split("\\n") if l.strip()]
    if lines and lines[0].isupper() and len(lines[0]) < 40:
        lines = lines[1:]  # section-header echo, e.g. "INDICATOR PHRASING"
    is_label = [l.lower().rstrip(":") in LANG_LABELS for l in lines]
    if not any(is_label):
        return "\\n".join(lines)
    i0 = is_label.index(True)
    n = 0
    while i0 + n < len(lines) and is_label[i0 + n]:
        n += 1
    head, rest = lines[:i0], lines[i0 + n:]
    if n == 1 and lines[i0].lower().rstrip(":") == "english":
        seg = []
        for k, l in enumerate(rest):
            if is_label[i0 + 1 + k]:
                break
            seg.append(l)
        return "\\n".join(head + seg)
    if n == 1:
        # single non-English label: texts follow one per language, English first
        return "\\n".join(head + rest[:1])
    per = len(rest) // n if rest and len(rest) % n == 0 else 1
    return "\\n".join(head + rest[:per])


def field_text(row, patterns):
    parts = []
    for field, v in row.items():
        if any(p in field.lower() for p in patterns) and isinstance(v, str) and v.strip():
            parts.append(english_only(v.strip()))
    return "\\n".join(parts)


for domain, df in frames.items():
    df["text_slug"] = df["slug"].map(text_slug)
    df["norm_title"] = df["title"].map(norm)
    df["wording"] = df.apply(lambda r: field_text(r, WORDING_PATTERNS), axis=1)
    df["norm_wording"] = df["wording"].map(norm)
    df["canonical"] = df.apply(lambda r: str(r["title"]) + "\\n" + field_text(r, SEMANTIC_PATTERNS), axis=1)

print("derived columns ready")
''')

code('''
# Embeddings over canonical text — all domains
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = {
    domain: model.encode(df["canonical"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    for domain, df in frames.items()
}
print({short(d): e.shape for d, e in emb.items()})
''')

code('''
# The ladder, ZOA -> each site. sims matrices are kept for the reverse
# (gap) direction later.
THRESH = 0.80

lookups = {}
for d in OTHERS:
    by_slug, by_textslug, by_title = {}, {}, {}
    for j, r in frames[d].iterrows():
        by_slug.setdefault(r["slug"], j)
        by_textslug.setdefault(r["text_slug"], j)
        if r["norm_title"]:
            by_title.setdefault(r["norm_title"], j)
    lookups[d] = (by_slug, by_textslug, by_title)

sims_fwd = {d: emb[ZOA] @ emb[d].T for d in OTHERS}

rows = []
for d in OTHERS:
    by_slug, by_textslug, by_title = lookups[d]
    sims = sims_fwd[d]
    best_j = sims.argmax(axis=1)
    for i, r in zoa_df.iterrows():
        j, tier = None, None
        if r["slug"] in by_slug:
            j, tier = by_slug[r["slug"]], 1
        elif r["text_slug"] in by_textslug:
            j, tier = by_textslug[r["text_slug"]], 2
        elif r["norm_title"] and r["norm_title"] in by_title:
            j, tier = by_title[r["norm_title"]], 2
        if j is None:
            j = int(best_j[i])
            tier = 3 if sims[i, j] >= THRESH else 0
        site_row = frames[d].loc[j]
        if tier == 0:
            cls = "absent"
        elif r["norm_wording"] and r["norm_wording"] == site_row["norm_wording"]:
            cls = "identical"
        else:
            cls = "modified"
        rows.append({
            "site": d, "zoa_idx": i, "zoa_title": r["title"], "zoa_sector": r["sector"],
            "tier": tier, "site_idx": j,
            "site_title": site_row["title"] if tier else None,
            "site_sector": site_row["sector"] if tier else None,
            "similarity": round(float(sims[i, j]), 3),
            "class": cls,
        })

matches = pd.DataFrame(rows)
matches.to_csv(OUT / "zoa_matches.csv", index=False)
matches.pivot_table(index="site", columns="class", aggfunc="size", fill_value=0).reindex(OTHERS)
''')

code('''
# Cutoff sensitivity: ECDF of each ZOA indicator's best embedding similarity,
# per site. A steep curve at the dashed line means that site's match count is
# fragile to the 0.80 cutoff choice; a flat one means it barely matters.
HIGHLIGHT = {BASE: C1, "adra_indikit_net": C2, "tearfund_indikit_net": C3, "tdh_indikit_net": "#008300"}

fig, ax = plt.subplots(figsize=(9, 4))
gray_labeled = False
for d in OTHERS:
    x = np.sort(sims_fwd[d].max(axis=1))
    yv = np.arange(1, len(x) + 1) / len(x)
    if d in HIGHLIGHT:
        ax.plot(x, yv, color=HIGHLIGHT[d], linewidth=2, label=short(d))
    else:
        ax.plot(x, yv, color=BASELINE, linewidth=1.1,
                label=None if gray_labeled else "other sites")
        gray_labeled = True
ax.axvline(THRESH, color=MUTED, linestyle="--", linewidth=1.2)
ax.text(THRESH + 0.006, 0.04, f"match cutoff = {THRESH}", color=INK2, fontsize=9)
ax.set_title("How close each ZOA indicator gets to its nearest counterpart, per site")
ax.set_xlabel("best cosine similarity of a ZOA indicator on the site")
ax.set_ylabel("share of ZOA indicators (ECDF)")
ax.legend(loc="upper left", labelcolor=INK2)
plt.tight_layout()
plt.show()
''')

md('''
## How much of ZOA's library each site shares

Of ZOA's 303 indicators, per site: shared **identically** (word-for-word English wording), shared but **modified**, or **absent**. The custom platforms (tdh, helvetas, savethechildren) can't score "identical" — no shared wording field — so read their bars as topical overlap.
''')

code('''
class_counts = (
    matches.pivot_table(index="site", columns="class", aggfunc="size", fill_value=0)
    .reindex(columns=["identical", "modified", "absent"], fill_value=0)
    .reindex(OTHERS)
)

order = class_counts[["identical", "modified"]].sum(axis=1).sort_values().index
cc = class_counts.loc[order]
fig, ax = plt.subplots(figsize=(9, 5))
left = np.zeros(len(cc))
for col, color in [("identical", C1), ("modified", C2), ("absent", GRID)]:
    ax.barh([short(d) for d in order], cc[col].to_numpy(float), left=left,
            height=0.62, color=color, edgecolor=SURFACE, linewidth=1.5, label=col)
    left += cc[col].to_numpy(float)
for y, shared in enumerate(cc[["identical", "modified"]].sum(axis=1)):
    ax.text(len(zoa_df) + 4, y, f"{shared} shared", va="center", color=INK2, fontsize=9)
ax.yaxis.grid(False)
ax.margins(x=0.16)
ax.set_title("ZOA's 303 indicators on each of the other sites")
ax.set_xlabel("ZOA indicators")
ax.legend(loc="lower right", labelcolor=INK2)
plt.tight_layout()
plt.show()

class_counts
''')

md('''
## How standard is each ZOA indicator?

Adoption count = on how many of the 9 other sites the indicator appears. High counts mark sector-standard indicators; zero marks **ZOA-only** indicators — ZOA's own footprint.
''')

code('''
coverage = (
    matches.assign(present=(matches["tier"] > 0).astype(int))
    .pivot_table(index="zoa_idx", columns="site", values="present")
    .reindex(range(len(zoa_df)), fill_value=0)
)
coverage.columns = [short(c) for c in coverage.columns]
coverage.insert(0, "zoa_title", zoa_df["title"])
coverage.insert(1, "zoa_sector", zoa_df["sector"])
coverage["adoption"] = coverage.drop(columns=["zoa_title", "zoa_sector"]).sum(axis=1).astype(int)
coverage.to_csv(OUT / "coverage_matrix.csv", index=False)

dist = coverage["adoption"].value_counts().sort_index().reindex(range(10), fill_value=0)
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar(dist.index, dist.to_numpy(), color=C1, edgecolor=SURFACE, linewidth=1.5, width=0.7)
for x, v in dist.items():
    if v:
        ax.text(x, v + 1.5, str(v), ha="center", color=INK2, fontsize=9)
ax.xaxis.grid(False)
ax.set_xticks(range(10))
ax.set_title("Adoption of ZOA's indicators across the other 9 sites")
ax.set_xlabel("number of other sites carrying the indicator")
ax.set_ylabel("ZOA indicators")
plt.tight_layout()
plt.show()

print("Most widely shared ZOA indicators:")
print(coverage.sort_values("adoption", ascending=False)[["zoa_title", "zoa_sector", "adoption"]]
      .head(10).to_string(index=False))
''')

code('''
# ZOA-only indicators — carried by no other site, not even the base
zoa_unique = coverage[coverage["adoption"] == 0][["zoa_title", "zoa_sector"]]
zoa_unique.to_csv(OUT / "zoa_unique.csv", index=False)
print(f"{len(zoa_unique)} ZOA-only indicators, by ZOA sector:")
print(zoa_unique["zoa_sector"].value_counts().to_string())
zoa_unique.head(15)
''')

md('''
## Which parts of ZOA's framework are sector-standard?

Per ZOA sector: the share of its indicators present on each site. Dark rows = sectors built from widely used indicators; light rows = sectors that are mostly ZOA's own construction.
''')

code('''
from matplotlib.colors import LinearSegmentedColormap

present = matches[matches["tier"] > 0]
sector_sizes = zoa_df["sector"].value_counts()
sector_cov = pd.DataFrame(index=sector_sizes.index, columns=OTHERS, dtype=float)
for d in OTHERS:
    got = present[present["site"] == d]
    got_sectors = zoa_df.loc[got["zoa_idx"].unique(), "sector"].value_counts()
    sector_cov[d] = (got_sectors.reindex(sector_sizes.index).fillna(0) / sector_sizes * 100).round(1)
sector_cov = sector_cov.loc[sector_cov.mean(axis=1).sort_values(ascending=False).index]

cmap = LinearSegmentedColormap.from_list("seq_blues", SEQ_BLUES)
fig, ax = plt.subplots(figsize=(9, 0.28 * len(sector_cov) + 1.8))
mesh = ax.pcolormesh(sector_cov.to_numpy(float), cmap=cmap, vmin=0, vmax=100,
                     edgecolors=SURFACE, linewidth=1.2)
ax.set_xticks(np.arange(len(sector_cov.columns)) + 0.5)
ax.set_xticklabels([short(d) for d in sector_cov.columns], rotation=30, ha="right")
ax.set_yticks(np.arange(len(sector_cov)) + 0.5)
ax.set_yticklabels([f"{s}  ({sector_sizes[s]})" for s in sector_cov.index], fontsize=8)
ax.invert_yaxis()
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Presence of each ZOA sector's indicators on the other sites")
cbar = fig.colorbar(mesh, ax=ax, shrink=0.55, pad=0.02)
cbar.set_label("% of ZOA sector present on site", color=INK2)
cbar.outline.set_visible(False)
plt.tight_layout()
plt.show()
''')

md('''
### How ZOA re-files the base's indicators

For confidently matched ZOA ↔ base pairs (tiers 1-2): which base sector each kept indicator came from, and which ZOA sector it now lives in. The base's fine-grained taxonomy gets consolidated into ZOA's much smaller one — the flows show which base sectors were merged where. Only the largest flows are drawn; the remainder is reported above the chart.
''')

code('''
import matplotlib.patches as mpatches
import matplotlib.path as mpath

pairs = matches[(matches["site"] == BASE) & matches["tier"].isin([1, 2])]
flows = (pairs.groupby(["site_sector", "zoa_sector"]).size().rename("n")
         .reset_index().sort_values("n", ascending=False))
TOP = 20
shown, rest = flows.head(TOP), flows.iloc[TOP:]
print(f"top {len(shown)} flows = {shown['n'].sum()} indicators; "
      f"{len(rest)} smaller flows ({rest['n'].sum()} indicators) not drawn")

lt = shown.groupby("site_sector")["n"].sum().sort_values(ascending=False)
rt = shown.groupby("zoa_sector")["n"].sum().sort_values(ascending=False)
GAP = 2.5


def stack(totals):
    pos, y = {}, 0.0
    for name, v in totals.items():
        pos[name] = y
        y += v + GAP
    return pos


lpos, rpos = stack(lt), stack(rt)
loff = dict.fromkeys(lt.index, 0.0)
roff = dict.fromkeys(rt.index, 0.0)
H = max(lt.sum() + GAP * (len(lt) - 1), rt.sum() + GAP * (len(rt) - 1))

fig, ax = plt.subplots(figsize=(10, max(5, 0.065 * H)))
Path = mpath.Path
CODES = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
         Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
for f in shown.sort_values(["site_sector", "n"], ascending=[True, False]).itertuples():
    y0 = lpos[f.site_sector] + loff[f.site_sector]
    y1 = rpos[f.zoa_sector] + roff[f.zoa_sector]
    loff[f.site_sector] += f.n
    roff[f.zoa_sector] += f.n
    verts = [(0.03, y0), (0.5, y0), (0.5, y1), (0.97, y1), (0.97, y1 + f.n),
             (0.5, y1 + f.n), (0.5, y0 + f.n), (0.03, y0 + f.n), (0.03, y0)]
    ax.add_patch(mpatches.PathPatch(Path(verts, CODES), facecolor=C1, alpha=0.3, edgecolor="none"))
for name, y in lpos.items():
    ax.add_patch(mpatches.Rectangle((0.0, y), 0.03, lt[name], facecolor="#256abf", edgecolor="none"))
    ax.text(-0.015, y + lt[name] / 2, f"{name[:42]}  ({lt[name]})",
            ha="right", va="center", fontsize=8, color=INK2)
for name, y in rpos.items():
    ax.add_patch(mpatches.Rectangle((0.97, y), 0.03, rt[name], facecolor="#256abf", edgecolor="none"))
    ax.text(1.015, y + rt[name] / 2, f"{name[:42]}  ({rt[name]})",
            ha="left", va="center", fontsize=8, color=INK2)
ax.text(0.0, -GAP, "base sector", ha="left", va="bottom", fontsize=9, color=MUTED)
ax.text(1.0, -GAP, "ZOA sector", ha="right", va="bottom", fontsize=9, color=MUTED)
ax.set_xlim(-0.62, 1.62)
ax.set_ylim(-GAP * 2, H + 1)
ax.invert_yaxis()
ax.axis("off")
ax.set_title("Where ZOA re-files the base's indicators (top flows)")
plt.tight_layout()
plt.show()
''')

md('''
## How ZOA's wording drifts

For confidently shared but **modified** pairs (tiers 1-2, both sides with wording text): character-level similarity of the English wording — 1.0 identical, ~0.9 small edits, below ~0.5 a rewrite.
''')

code('''
from difflib import SequenceMatcher

mod = matches[(matches["class"] == "modified") & matches["tier"].isin([1, 2])].copy()
mod["site_norm_wording"] = [frames[m.site].loc[m.site_idx, "norm_wording"] for m in mod.itertuples()]
mod["zoa_norm_wording"] = zoa_df.loc[mod["zoa_idx"], "norm_wording"].to_numpy()
mod = mod[(mod["zoa_norm_wording"] != "") & (mod["site_norm_wording"] != "")]
mod["wording_sim"] = [
    round(SequenceMatcher(None, a, b).ratio(), 3)
    for a, b in zip(mod["zoa_norm_wording"], mod["site_norm_wording"])
]
print(mod.groupby("site")["wording_sim"].agg(pairs="count", median="median").round(2))

# strip plot: one dot per pair — honest about small samples (a pooled
# histogram would hide that some sites contribute only a handful of pairs)
sites_present = [d for d in OTHERS if (mod["site"] == d).any()]
rng = np.random.default_rng(7)
fig, ax = plt.subplots(figsize=(9, 0.55 * len(sites_present) + 1.4))
for k, d in enumerate(sites_present):
    v = mod.loc[mod["site"] == d, "wording_sim"].to_numpy()
    ax.scatter(v, np.full(len(v), float(k)) + rng.uniform(-0.14, 0.14, len(v)),
               s=26, color=C1, alpha=0.65, edgecolors=SURFACE, linewidths=0.5)
    ax.text(1.04, k, f"n={len(v)}", va="center", color=INK2, fontsize=9)
ax.set_yticks(range(len(sites_present)))
ax.set_yticklabels([short(d) for d in sites_present])
ax.yaxis.grid(False)
ax.set_xlim(0, 1.12)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_title("Wording drift of shared-but-modified indicators")
ax.set_xlabel("wording similarity to ZOA (1.0 = identical)")
plt.tight_layout()
plt.show()

print("\\nExamples — same indicator, meaningfully different wording:")
for _, m in mod[mod["wording_sim"].between(0.4, 0.75)].sort_values("wording_sim").head(3).iterrows():
    print("=" * 90)
    print(f"{m['zoa_title']}  vs {short(m['site'])}  (wording similarity {m['wording_sim']})")
    print(f"  ZOA : {zoa_df.loc[m['zoa_idx'], 'wording'][:280]}")
    print(f"  SITE: {frames[m['site']].loc[m['site_idx'], 'wording'][:280]}")
''')

md('''
## Semantic comparison

Meaning-level view on top of the ladder, carried over from `data_analysis.ipynb`. First the **mutual best-match table**: for each ZOA indicator its closest embedding match anywhere in the other libraries, with `mutual = True` when each is the other's best match — the strongest "probably the same indicator" candidates even where slug/title matching fails (the only channel for the custom platforms). Then **free-text semantic search** across all ten libraries at once.

Caveat: embeddings measure *topical* similarity, not measurement equivalence — treat high-scoring pairs as candidates to eyeball, not confirmed matches.
''')

code('''
# Mutual best matches: ZOA vs the concatenated corpus of all other libraries
corpus_others = pd.concat(
    [frames[d][["title", "sector"]].assign(site=d) for d in OTHERS],
    ignore_index=True,
)
emb_others = np.vstack([emb[d] for d in OTHERS])

sim_all = emb[ZOA] @ emb_others.T          # 303 x n_others
best_other = sim_all.argmax(axis=1)
best_zoa = sim_all.argmax(axis=0)

semantic = pd.DataFrame(
    {
        "zoa_title": zoa_df.loc[i, "title"],
        "zoa_sector": zoa_df.loc[i, "sector"],
        "match_title": corpus_others.loc[j, "title"],
        "match_site": short(corpus_others.loc[j, "site"]),
        "match_sector": corpus_others.loc[j, "sector"],
        "similarity": round(float(sim_all[i, j]), 3),
        "mutual": bool(best_zoa[j] == i),
    }
    for i, j in enumerate(best_other)
).sort_values("similarity", ascending=False).reset_index(drop=True)

semantic.to_csv(OUT / "semantic_matches.csv", index=False)
print(f"mutual top-1 pairs: {semantic['mutual'].sum()} of {len(semantic)}")
semantic.head(20)
''')

code('''
# Free-text semantic search across all ten libraries
corpus_all = pd.concat(
    [frames[d][["title", "sector"]].assign(site=short(d)) for d in [ZOA] + OTHERS],
    ignore_index=True,
)
emb_all = np.vstack([emb[d] for d in [ZOA] + OTHERS])


def search(query, top_k=8):
    q = model.encode([query], normalize_embeddings=True)
    scores = (emb_all @ q.T).ravel()
    idx = scores.argsort()[::-1][:top_k]
    result = corpus_all.loc[idx, ["site", "title", "sector"]].copy()
    result.insert(0, "score", scores[idx].round(3))
    return result.reset_index(drop=True)


search("resilience of smallholder farmers to drought")
''')

code('''
search("safe drinking water access in rural communities")
''')

md('''
## What the others measure that ZOA doesn't

Reverse direction: each site's indicators with no ZOA counterpart (same ladder, site → ZOA). The per-site counts show how much of their library lies outside ZOA's scope and in which of *their* sectors; the base's uncovered indicators are the canonical candidate list for additions.
''')

code('''
# reverse ladder: site indicator -> any ZOA counterpart?
zoa_by_slug, zoa_by_textslug, zoa_by_title = {}, {}, {}
for j, r in zoa_df.iterrows():
    zoa_by_slug.setdefault(r["slug"], j)
    zoa_by_textslug.setdefault(r["text_slug"], j)
    if r["norm_title"]:
        zoa_by_title.setdefault(r["norm_title"], j)

gap_rows = []
for d in OTHERS:
    df = frames[d]
    best = sims_fwd[d].max(axis=0)  # per site indicator: best cosine vs any ZOA indicator
    covered = (
        df["slug"].isin(zoa_by_slug)
        | df["text_slug"].isin(zoa_by_textslug)
        | df["norm_title"].isin(zoa_by_title)
        | (best >= THRESH)
    )
    gaps = df[~covered]
    top = gaps["sector"].value_counts().head(3)
    tops = ", ".join(f"{s} ({n})" for s, n in top.items())
    gap_rows.append({"site": short(d), "indicators": len(df), "not_in_zoa": len(gaps),
                     "not_in_zoa_pct": round(100 * len(gaps) / len(df), 1)})
    print(f"{short(d):16s} {len(gaps):4d}/{len(df):4d} not in ZOA | top sectors: {tops}")
    if d == BASE:
        base_gaps = gaps

gaps_overview = pd.DataFrame(gap_rows).set_index("site")

base_gaps[["title", "sector"]].to_csv(OUT / "gaps_base.csv", index=False)
print(f"\\nBase indicators without a ZOA counterpart, by base sector (top 15 of {len(base_gaps)}):")
print(base_gaps["sector"].value_counts().head(15).to_string())
''')

md('''
## Deep dives: Tearfund, WHH and Save the Children

Three sites at increasing distance from ZOA get the same in-depth battery:

- **Tearfund** — same IndiKit engine, yet almost none of ZOA's library appears on it. Different themes, or the same themes measured with differently constructed indicators?
- **WHH** — same engine, mid-range overlap, thematically close to ZOA (food security / WASH). Where exactly does the overlap stop?
- **Save the Children** — entirely custom platform: no slug/title channel, so everything is embedding-based *topical* similarity; their wording comes from the `Definition` field and "identical" is impossible by construction.

The battery per site: ① full overlap inventory, ② near-misses (0.60–0.80 — same topic, different construction), ③ topical sector coverage in both directions, ④ a t-SNE topical map (base gray, ZOA blue, site yellow; qualitative only — the projection distorts distances), ⑤ measurement-style comparison (unit type + subject, regex over the full libraries), ⑥ the site's own territory (best similarity to ZOA < 0.6).
''')

code('''
# Shared deep-dive helpers, parameterized by site
from sklearn.manifold import TSNE

TF, WHH, SCI = "tearfund_indikit_net", "whh_indikit_net", "indicators_savethechildren_net"


def display_wording(df):
    # savethechildren has no wording/phrasing field — fall back to its Definition
    w = df["wording"]
    if (w == "").all():
        w = df.apply(lambda r: field_text(r, ("definition",)), axis=1)
    return w


def dd_inventory(site):
    pairs = matches[(matches["site"] == site) & (matches["tier"] > 0)]
    print(f"ZOA indicators with a {short(site)} counterpart: {len(pairs)} of {len(zoa_df)}")
    return (pairs[["zoa_title", "zoa_sector", "site_title", "site_sector", "tier", "similarity", "class"]]
            .sort_values("similarity", ascending=False).reset_index(drop=True))


def dd_near_misses(site, lo=0.60):
    df, sims = frames[site], sims_fwd[site]
    best_j, best = sims.argmax(axis=1), sims.max(axis=1)
    matched_idx = set(matches[(matches["site"] == site) & (matches["tier"] > 0)]["zoa_idx"])
    site_w = display_wording(df)
    near = pd.DataFrame({
        "zoa_title": zoa_df["title"], "zoa_sector": zoa_df["sector"],
        "site_title": df.loc[best_j, "title"].to_numpy(),
        "site_sector": df.loc[best_j, "sector"].to_numpy(),
        "similarity": best.round(3),
        "zoa_wording": zoa_df["wording"].str.slice(0, 150),
        "site_wording": site_w.loc[best_j].str.slice(0, 150).to_numpy(),
    })
    near = near[(near["similarity"] >= lo) & (near["similarity"] < THRESH)
                & (~near.index.isin(matched_idx))].sort_values("similarity", ascending=False)
    near.to_csv(OUT / f"{short(site)}_near_misses.csv", index=False)
    print(f"{len(near)} near-misses ({lo} <= similarity < {THRESH})")
    return near


def dd_sector_coverage(site):
    # topical_pct = share of the sector's indicators with best similarity >= 0.6
    # matched_pct = share above the 0.80 match cutoff
    def one_direction(df, best_sims):
        return (pd.DataFrame({"sector": df["sector"], "best": best_sims})
                .groupby("sector")["best"]
                .agg(n="count",
                     topical_pct=lambda s: round(100 * (s >= 0.6).mean(), 1),
                     matched_pct=lambda s: round(100 * (s >= THRESH).mean(), 1))
                .sort_values("topical_pct", ascending=False))

    sims = sims_fwd[site]
    print(f"ZOA sectors — how far {short(site)} covers them topically:")
    print(one_direction(zoa_df, sims.max(axis=1)).to_string())
    print(f"\\n{short(site)} sectors — how far ZOA covers them topically:")
    print(one_direction(frames[site], sims.max(axis=0)).to_string())


def dd_tsne(site):
    df = frames[site]
    pts = np.vstack([emb[BASE], emb[ZOA], emb[site]])
    xy = TSNE(n_components=2, random_state=0, init="pca", perplexity=30).fit_transform(pts)
    n_base, n_zoa = len(frames[BASE]), len(zoa_df)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(xy[:n_base, 0], xy[:n_base, 1], s=12, color=BASELINE, alpha=0.55,
               edgecolors="none", label=f"base ({n_base})")
    ax.scatter(xy[n_base:n_base + n_zoa, 0], xy[n_base:n_base + n_zoa, 1], s=20, color=C1,
               alpha=0.8, edgecolors="none", label=f"zoa ({n_zoa})")
    ax.scatter(xy[n_base + n_zoa:, 0], xy[n_base + n_zoa:, 1], s=20, color=C3,
               alpha=0.8, edgecolors="none", label=f"{short(site)} ({len(df)})")
    # label the biggest sectors of ZOA and the site at their centroids
    for df_, sl in [(zoa_df, slice(n_base, n_base + n_zoa)), (df, slice(n_base + n_zoa, None))]:
        sub = xy[sl]
        for sector in df_["sector"].value_counts().head(3).index:
            m = (df_["sector"] == sector).to_numpy()
            ax.annotate(str(sector)[:34], (sub[m, 0].mean(), sub[m, 1].mean()),
                        color=INK, fontsize=8, ha="center", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Topical map of base, ZOA and {short(site)} indicators (t-SNE)")
    ax.legend(loc="upper right", labelcolor=INK2)
    plt.tight_layout()
    plt.show()


def unit_type(w):
    w = re.sub(r"^the ", "", str(w).strip().lower())
    if w.startswith("%") or w.startswith("percentage") or w.startswith("percent"):
        return "% of ..."
    if w.startswith(("number", "no.", "#", "total number")):
        return "number of ..."
    if w.startswith(("average", "mean", "median")) or "score" in w[:40]:
        return "score / average"
    return "other"


SUBJECTS = ["household", "individual", "child", "farmer", "wom", "communit", "people", "beneficiar"]


def subject(w):
    w = str(w).lower()
    for s in SUBJECTS:
        if s in w[:120]:
            return s
    return "none of these"


def dd_style(site):
    site_w = display_wording(frames[site])
    style = pd.DataFrame({
        "library": ["zoa"] * len(zoa_df) + [short(site)] * len(site_w),
        "unit": [unit_type(w) for w in list(zoa_df["wording"]) + list(site_w)],
        "subject": [subject(w) for w in list(zoa_df["wording"]) + list(site_w)],
    })
    print("Unit type (% of each library):")
    print((pd.crosstab(style["library"], style["unit"], normalize="index") * 100).round(1).to_string())
    print("\\nFirst subject mentioned in the wording (% of each library):")
    print((pd.crosstab(style["library"], style["subject"], normalize="index") * 100).round(1).to_string())


def dd_only(site, lo=0.6):
    df = frames[site]
    best = sims_fwd[site].max(axis=0)
    only = df.loc[best < lo, ["title", "sector"]]
    only.to_csv(OUT / f"{short(site)}_only.csv", index=False)
    print(f"{len(only)} of {len(df)} {short(site)} indicators have no topical ZOA counterpart "
          f"(best < {lo}), by sector:")
    print(only["sector"].value_counts().to_string())
    return only


print("deep-dive helpers ready")
''')

md('''
### Tearfund vs ZOA
''')

code('''
dd_inventory(TF)
''')

code('''
dd_near_misses(TF).head(12)
''')

code('''
dd_sector_coverage(TF)
''')

code('''
dd_tsne(TF)
''')

code('''
dd_style(TF)
''')

code('''
dd_only(TF).head(12)
''')

md('''
### WHH vs ZOA
''')

code('''
dd_inventory(WHH)
''')

code('''
dd_near_misses(WHH).head(12)
''')

code('''
dd_sector_coverage(WHH)
''')

code('''
dd_tsne(WHH)
''')

code('''
dd_style(WHH)
''')

code('''
dd_only(WHH).head(12)
''')

md('''
### Save the Children vs ZOA

Reminder: custom platform — all matches here are embedding-based topical similarity, wording comes from their `Definition` field, and "identical" cannot occur.
''')

code('''
dd_inventory(SCI)
''')

code('''
dd_near_misses(SCI).head(12)
''')

code('''
dd_sector_coverage(SCI)
''')

code('''
dd_tsne(SCI)
''')

code('''
dd_style(SCI)
''')

code('''
dd_only(SCI).head(12)
''')

md('''
### Deep-dive conclusions

*(filled in from the actual numbers after the run — one short verdict per site: where its overlap with ZOA stops, and why)*
''')

md('''
## Summary

*(interpretation filled in below the final table)*
''')

code('''
summary = class_counts.copy()
summary["shared"] = summary["identical"] + summary["modified"]
summary["zoa_covered_pct"] = (100 * summary["shared"] / len(zoa_df)).round(1)
summary = summary.join(gaps_overview[["indicators", "not_in_zoa", "not_in_zoa_pct"]]
                       .set_axis(OTHERS))
summary.index = [short(d) for d in summary.index]
summary.to_csv(OUT / "summary.csv")

# dumbbell: two dots per site — the line length is the asymmetry between
# "how much of ZOA they carry" and "how much of theirs lies outside ZOA"
s = summary.sort_values("zoa_covered_pct")
y = np.arange(len(s))
fig, ax = plt.subplots(figsize=(9, 5))
for yi, (a, b) in enumerate(zip(s["zoa_covered_pct"], s["not_in_zoa_pct"])):
    ax.plot([a, b], [yi, yi], color=GRID, linewidth=2.5, zorder=1)
ax.scatter(s["zoa_covered_pct"], y, s=80, color=C1, zorder=2,
           edgecolors=SURFACE, linewidths=1, label="% of ZOA's library on the site")
ax.scatter(s["not_in_zoa_pct"], y, s=80, color=C2, zorder=2,
           edgecolors=SURFACE, linewidths=1, label="% of the site's library not in ZOA")
ax.set_yticks(y)
ax.set_yticklabels(s.index)
ax.yaxis.grid(False)
ax.set_xlim(-3, 103)
ax.set_title("Overlap between ZOA and each site, both directions", pad=34)
ax.set_xlabel("%")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncols=2, labelcolor=INK2)
plt.tight_layout()
plt.show()

summary
''')

nb.cells = cells
nbf.write(nb, "zoa_comparison.ipynb")
print(f"wrote zoa_comparison.ipynb with {len(cells)} cells")
