# Plan — `zoa_comparison.ipynb`: ZOA vs all other IndiKit sites

## Goal

`base_comparison.ipynb` measured every site against the base (www.indikit.net). This notebook flips the reference point: **ZOA's library (303 indicators, 30 sectors) is the anchor**, compared against the other 9 sites from `files/links` (the base + adra, irw, tearfund, whh, winrock, tdh, helvetas, savethechildren). Questions it answers:

1. **Which of ZOA's indicators do other organizations also use?** Per site: shared identically, shared but reworded, or absent.
2. **How "standard" is each ZOA indicator?** An adoption count per indicator (on how many of the 9 sites it appears) — separating sector-standard indicators from ZOA-only ones.
3. **Where is ZOA unique?** Indicators no other site carries — ZOA's own footprint.
4. **What is ZOA missing?** Indicators other sites (especially the base) carry that have no ZOA counterpart — candidate additions, grouped by sector.
5. **How does ZOA's wording drift** from the sites that share its indicators?

## Data & method (reused from `base_comparison.ipynb`)

- Input: scraped snapshots in `data/*/indicators.json` — no scraping.
- Multilingual wording sections are reduced to English first (`english_only`, handles the three layouts found earlier: label-block, interleaved, tab-residue).
- **Matching ladder** per (ZOA indicator, other site): ① exact slug → ② slug-text or exact normalized title → ③ embedding cosine ≥ 0.80 (`all-MiniLM-L6-v2` over wording/phrasing/purpose/definition fields). Below cutoff = not present on that site.
- Classification of matches: **identical** (English wording char-identical after normalization) vs **modified**. Custom-platform sites (helvetas, savethechildren, tdh) can only match via embeddings — topical similarity, flagged as such.

## Notebook sections

1. **Load & overview** — all 10 datasets, ZOA as reference; per-site size/template table.
2. **Match ZOA against each site** — ladder above; long match table (`zoa_title × site → tier, similarity, class`) and a 303 × 9 coverage matrix. Includes a **cutoff-sensitivity ECDF chart** (best similarity per ZOA indicator, per site, with the 0.80 cutoff marked): a steep curve near the cutoff means that site's match count is fragile to the threshold choice.
3. **Per-site sharing chart** — stacked bars per site: identical / modified / absent (of ZOA's 303).
4. **Adoption count** — distribution of "carried by N other sites" per ZOA indicator; top most-widely-shared indicators; the ZOA-unique list (on no other site, not even the base).
5. **Sector coverage heatmap** — ZOA sector × site: % of that sector's ZOA indicators present on the site. Shows which parts of ZOA's framework are sector-standard vs house-made. Followed by an **alluvial flow chart** (hand-rolled matplotlib, no new dependency) showing how ZOA re-files the base's indicators into its own sector taxonomy — top ~20 flows, smaller ones reported as a count.
6. **Wording drift** — for shared-but-modified pairs: **strip plot** per site (one dot per indicator pair — honest about small samples, unlike a pooled histogram) + a few side-by-side examples.
7. **Semantic comparison** (carried over from `data_analysis.ipynb`) — meaning-level view on top of the ladder:
   - **Mutual best-match table**: for each ZOA indicator, its best embedding match across all other libraries, with a `mutual` flag (each is the other's best match) — the strongest "probably the same indicator" candidates even where slug/title matching fails.
   - **Free-text semantic search**: embed a query and retrieve the nearest indicators from ZOA *and* the other libraries side by side (example queries included) — useful for checking how different organizations measure a given topic.
8. **Gaps** — reverse direction: per site, how much of their library has no ZOA counterpart and in which of their sectors; detailed table for the base (canonical candidates ZOA could adopt).
9. **Site deep dives: Tearfund, WHH, Save the Children** — see the dedicated section below.
10. **Summary** — per-site table + **dumbbell chart** (two connected dots per site: % of ZOA's library on the site vs % of the site's library not in ZOA — shows the asymmetry gap directly, cleaner than grouped bars) + written interpretation (filled in from the actual numbers).

## Site deep dives (inserted after "What the others measure that ZOA doesn't")

Three sites at increasing distance from ZOA get the same in-depth battery, implemented once as a **shared helper cell** and run per site:

- **Tearfund** — same IndiKit engine, yet only ~4 of ZOA's indicators appear identically and ~263 of 303 are absent. Different themes, or same themes measured differently?
- **WHH** — same engine, mid-range overlap, thematically close to ZOA (food security / WASH). Where exactly does the overlap stop?
- **Save the Children** — entirely custom platform (PRIME framework): no slug/title channel, so *everything* is embedding-based topical similarity; their wording is drawn from the `Definition` field, and "identical" is impossible by construction. Findings read as thematic proximity, not shared indicators.

The battery, per site (helper functions, so each site section is a handful of one-line cells):

1. **Overlap inventory** — the complete list of ZOA↔site matched pairs (all tiers) with class and similarity: everything the two libraries demonstrably share.
2. **Near-miss table** — pairs with embedding similarity in the 0.60–0.80 band (below the match cutoff): same topic, differently built indicator. Shown side by side (ZOA wording vs site wording). This is the direct evidence for "adapted to their needs" vs "absent".
3. **Topical coverage, both directions** — per ZOA sector: the best site similarity of each indicator (how much of that ZOA sector the site covers *topically*, matched or not); and per site sector the same against ZOA. Separates sectors genuinely out of the other's scope from sectors both cover but word differently.
4. **Topical map (t-SNE embedding scatter)** — all base + ZOA + site indicators projected to 2D: base in gray, ZOA blue, the site yellow (same role-color on every map), with the biggest sectors of each labeled at their centroids. Site clusters with no ZOA dots nearby = genuinely different themes; interleaved regions = same themes, different indicators. Qualitative only (projections distort distances).
5. **Measurement-style comparison** — classify each library's wording by unit type (`% of …` vs `number of …` vs score/average) and first subject mentioned (households / individuals / children / communities …) with simple regexes, over the full libraries. Quantifies systematic house-style choices in aggregate rather than anecdote.
6. **Site-only territory** — the site's indicators with no topical ZOA counterpart (best similarity < 0.6), grouped by the site's sectors: its genuinely distinct measurement territory.
7. **Markdown conclusion** — one combined section answering "why does the overlap stop where it does" per site, based on the actual numbers (filled in after the run).

Extra outputs per site: `comparison_zoa/<site>_near_misses.csv`, `comparison_zoa/<site>_only.csv`.

## Outputs

`comparison_zoa/`: `zoa_matches.csv` (long match table), `coverage_matrix.csv`, `zoa_unique.csv`, `semantic_matches.csv` (mutual best-match table), `gaps_base.csv`, `summary.csv`.

## Verification

Execute end-to-end with nbconvert on the project venv; sanity checks: ZOA×base results must be consistent with `base_comparison.ipynb` (~196 tier-2, ~179 identical); custom platforms show tier-3 matches only; all CSVs written; charts render without label collisions.
