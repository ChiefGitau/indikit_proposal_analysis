# ZOA-IndiKit Indicator Profiler

A Streamlit app that analyses and visualises the ZOA-IndiKit export
(`data/Export indikit.csv`, 386 indicators, latin-1 encoded). It serves two
audiences:

- **MEAL staff** — taxonomy, coverage and data-quality views that answer
  "what does IndiKit contain and where are the gaps?".
- **Developers** — text-profile, semantic-map and duplicate views that inform
  the matching pipeline (`coding_plan.md`).

## Pages

| Page | What it shows |
|------|---------------|
| Landing | Global filters (Section / Sector / Level), cleaned table, CSV export |
| 1 · Taxonomy | KPIs, sunburst, treemap, per-sector bar with drill-down |
| 2 · Coverage | Sector×Level and Sector×Sub-Sector heatmaps, gap callouts |
| 3 · Text Profile | Wording length, quantifier patterns, n-grams, word cloud, search |
| 4 · Semantic Map | UMAP of indicator embeddings, coloured by sector/level/sub-sector |
| 5 · Duplicates | Near-duplicate pairs above a similarity threshold, CSV export |
| 6 · Data Quality | Missing values, level values, language coverage, URL check |
| 7 · Analysis Log | Every section's collected tables + metrics, browse + export as one zip |
| 8 · Proposal Overview | Per-proposal metadata + Africa map (dummy data for now) |
| 9 · Site Comparison | Scrapes the live IndiKit site (BeautifulSoup) and diffs it against the CSV: coverage (removed indicators), per-field text drift, and whether the site can fill the CSV's missing indicator levels |

### Analysis log

Each analysis page writes the tables and metrics it computes to a durable log
under `data/analysis_log/` (one CSV per table + a `manifest.json`). The
**Analysis Log** page reads it all back so findings can be browsed in one place
and downloaded as a single zip — across sessions, regardless of which pages were
open. Each entry is tagged with the filters that produced it; re-running a page
refreshes that section. Delete the folder (or use the page's "Clear log" button)
to reset.

### Site comparison (live scrape)

The **Site Comparison** page fetches the live site's sitemap and scrapes each
`/indicator/` page with BeautifulSoup, honouring the site's `Crawl-Delay: 1`
(so a full ~300-page crawl takes ≈5 minutes). Results are cached to
`data/indicators/data_analysis/site_scrape/scraped.json`, so you only pay the
crawl once — reruns read the cache. Use the "Only missing / not-yet-cached"
mode to top up after a partial or interrupted crawl. It reports which CSV
indicators were removed from the site and, per surviving indicator, the
similarity (0–1) between the CSV text and the live page for name, EN/FR wording
and purpose.

It also recovers each indicator's **level** (Output / Outcome / Impact /
Organisational): the detail pages don't state it, but the sector listings can be
filtered by level server-side (`?filterValues[]=<id>`), so the page reads the
level(s) from there and reports whether the site can fill the levels the CSV is
missing (as of this export it can't — those indicators are uncategorised on the
site too).

## Run

```bash
cd zoa_indikit_app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The filters in the sidebar apply across every page. **All six pages work with
just the core dependencies** — the Semantic Map and Duplicates pages
automatically fall back to a lexical TF-IDF + PCA path when the optional ML
libraries are absent (they tell you which path is active).

### Optional: real semantic embeddings

For meaning-based (rather than word-overlap) similarity, install the optional
extras. `umap-learn` compiles `llvmlite`, which needs CMake:

```bash
brew install cmake                 # macOS; or your package manager
pip install sentence-transformers umap-learn wordcloud
```

First load of the Semantic Map / Duplicates pages then downloads the
`all-MiniLM-L6-v2` model and computes embeddings (cached to
`data/embeddings.npy` thereafter). `umap-learn` is optional even here — without
it the projection uses PCA.

## Refreshing the data

When `Export indikit.csv` changes, delete the cached embeddings so they are
recomputed:

```bash
rm -f data/embeddings.npy
```

(The Streamlit `@st.cache_data` loader cache also clears on a server restart or
via the "Clear cache" item in the app's ☰ menu.)
