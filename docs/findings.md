# Findings

What the project found, from the IndiKit metadata build (Phase 1), the document preprocessing (Phase 2), the LLM
matching pipeline (Phase 3) and the first full run of the portfolio. Numbers come from `outputs/latest/`
(run `run-20261115-233434`), the phase reports in `docs/` and the run's CSVs. Written 2026-11-16, updated 2026-11-24
after a re-scrape of zoa.indikit.net (Section 2a).

The headline numbers come with one caveat that applies to everything below: **the matches have not been checked against
a verified gold set** (see Section 9). They are the model's answers, not confirmed facts.

## 1. Headline

- **115 documents** in the portfolio; 108 were queried, 7 only point to another document for their indicators.
- **1 236 document indicators** were found; **362 (29.3 %) match an IndiKit indicator**, 73 verbatim and 289 close.
- **146 of the 351 IndiKit indicators (42 %) are used** in at least one document; **205 are never used**.
- **874 document indicators have no IndiKit equivalent.** They form 605 clusters, 105 of which recur in more than
  one document.
- The full run cost **$24.25** in 8 086 LLM calls (about $0.22 per document) with the OpenAI GPT-6 family.
- Six copies of one proposal sit in six country folders and inflate every cross-country count (Section 4).
- **The IndiKit website lost 97 indicator pages between 16 and 24 November 2026**, almost all Food Security and
  Livelihoods. Only 207 of the 351 library indicators are on the site today, and 46 indicators that proposals use are
  not among them (Section 2a).

## 2. The IndiKit library

From `docs/phase_1_report.md`:

- The library has **351 indicators**: 344 in the Excel export plus 7 PIP indicators (5889-5895) that exist only on
  the site. When the library was built (scrape of 16 November 2026) 303 were both in the export and on the site and
  **41 were export-only** (`on_site = false`), not visible on the IndiKit website. The site has changed since; see 2a.
- Export and site **agree on every name, wording and purpose** after normalisation; there were no merge conflicts.
- Data gaps: 18 records have no English wording (5806-5814, 5885, 5886 and the 7 PIP pages), 49 have no level and
  154 have no disaggregation topic. The PIP pages carry a name and purpose but no wording block, so the pipeline
  matches them on the name.
- The library is **lopsided towards Food Security and Livelihoods** (172 of 351). WASH has 56, CVA 32, Gender
  Diversity & Inclusion 24, Peacebuilding 23, Shelter 21, Climate Resilience 17, Localisation 3, Peace Nexus 3.
- The brief's description of the repository was out of date in several places (file sizes, duplicated code blocks,
  file names); the discrepancies are listed in the phase 1 report, Section 5.

### 2a. The website changed during the project

A second scrape on 2026-11-24 (every library ID plus the sitemap, `scraped.json`; the November scrape is kept as
`scraped_2026-11-16.json`) found:

| | 16 Nov 2026 | 24 Nov 2026 |
|---|---|---|
| indicator pages that load | 303 (+7 PIP) | **207** |
| IDs in the site's sitemap | 214 | 214 |
| library indicators not on the site | 41 | **144** |

- **97 pages that loaded in November return 404 now**, 94 of them Food Security and Livelihoods (for example Months
  of Adequate Household Food Provisioning 5092, Accessing a Service 5532, Active Participation in Activities 5530).
  The sitemap never listed them, so the removal looks planned rather than accidental. Seven further IDs are in the
  sitemap but have no page.
- **Food Security and Livelihoods shrank from 172 to 33 live indicators.** The live library is now led by WASH (55),
  Cash and Voucher Assistance (32), Gender Diversity & Inclusion (23), Peacebuilding (22) and Shelter (21).
- **46 indicators that proposals use are no longer on the website**, with 113 matches in 51 documents. Gender
  Representation (5428, 17 proposals), Number of Cash Transfer Recipients (5239, 11), Active Participation in
  Activities (5530, 10) and Accessing a Service (5532, 9) are the most used of them.
- Of the 207 live indicators, **100 are used and 107 are not**. Six live sub-sectors have no indicator in use.
- The matches themselves did not change: the run asked about all 351 indicators. Only the `on_site` / `export_only`
  flags in the outputs, the Analysis IndiKit page, the Excel exports and the deck were refreshed
  (`python -m pipeline.run refresh-site`, also triggered from the Site Comparison page after a full re-scrape).

## 3. The document portfolio

- **115 documents** across nine countries: South Sudan 20, Uganda 19, Sudan 17, Burundi 14, Syria 13, Ethiopia 11,
  Yemen 10, DRC 9, Burkina Faso 2. File types: 75 docx, 33 pdf, 7 xlsx. Language: 112 English, 3 French.
- **Parsing works for almost everything**: 107 documents parsed fully, 8 partially. The partial ones are scanned
  PDFs without a text layer, mostly copies of the same "We are able" proposal; they would need OCR.
- **67 documents contain logframe content, 48 do not.** Narrative proposals without a logframe produce few or no
  indicators (Section 6).
- **40 documents refer to another file for their indicators.** 7 have no logframe of their own (flag
  `indicators_elsewhere`) and were reported but not queried; 33 have a logframe and also point to an annex or tracker
  (`indicators_also_elsewhere`), so their indicators may be counted twice across the portfolio.
- **One proposal is duplicated six times.** `ZOA-we-are-able-full-proposal-v6.pdf` sits in the Burundi, DRC,
  Ethiopia, South Sudan, Sudan and Uganda folders with identical content. Without the five extra copies the portfolio
  is 110 unique documents, 1 045 indicators and 338 matches (32.3 %).

## 4. Matching results

### Overall

| | indicators | share |
|---|---|---|
| verbatim | 73 | 5.9 % |
| close | 289 | 23.4 % |
| none | 874 | 70.7 % |

- Most matches are **close rather than verbatim**: proposals adapt IndiKit wording to the project rather than copy it.
- Per document the picture is better than the pooled figure suggests. The **median document has 44.7 % of its
  indicators matched**, 14 documents reach 100 % and only one document with indicators has none matched. The pooled
  rate is pulled down by a few large logframes with many project-specific indicators.
- **327 rows are flagged for review**, almost all because a close match is by definition a judgement call (289), plus
  28 with alternative candidates and 10 others.
- **111 matched rows point to IndiKit indicators that are not on the website today** (43 before the November
  removals); see Section 2a.

### By country

| country | documents | indicators | matched | share |
|---|---|---|---|---|
| Syria | 13 | 98 | 52 | 53.1 % |
| Burkina Faso | 2 | 7 | 3 | 42.9 % |
| South Sudan | 20 | 158 | 58 | 36.7 % |
| Burundi | 14 | 151 | 52 | 34.4 % |
| Yemen | 10 | 172 | 54 | 31.4 % |
| Ethiopia | 11 | 210 | 52 | 24.8 % |
| Uganda | 19 | 108 | 26 | 24.1 % |
| DRC | 9 | 192 | 38 | 19.8 % |
| Sudan | 17 | 140 | 27 | 19.3 % |

- **Syria uses IndiKit most**, with 23 of the 73 verbatim matches in the whole portfolio. DRC and Sudan use it least.
- The We-are-able copies add about 38 indicators, of which only 5 match, to each of six countries, lowering their shares.

### By sector

| sector | documents | IndiKit indicators used | of | matches |
|---|---|---|---|---|
| Food Security and Livelihoods | 61 | 65 | 172 | 206 |
| Water, Sanitation and Hygiene | 29 | 21 | 56 | 64 |
| Peacebuilding | 22 | 12 | 23 | 43 |
| Shelter | 25 | 13 | 21 | 42 |
| Cash and Voucher Assistance | 17 | 17 | 32 | 37 |
| Gender Diversity & Inclusion | 14 | 7 | 24 | 21 |
| Climate Resilience | 9 | 9 | 17 | 16 |
| Localisation | 2 | 1 | 3 | 2 |
| Peace Nexus | 1 | 1 | 3 | 1 |

- Food Security and Livelihoods dominates in matches but **only 65 of its 172 indicators are ever used**; 107 of the
  205 never-used indicators belong to it.
- **Shelter and Peacebuilding are used most intensively** relative to their size (13 of 21 and 12 of 23 indicators).
- **Climate Resilience matches are almost all close** (1 verbatim of 16): proposals describe climate outcomes in their
  own words.

### Most-used IndiKit indicators

The top of the list: Gender Representation (5428) in 17 documents, Adoption of Promoted Practices (5490) in 14,
PIP households implementing their plans (5889) in 13, Food Consumption Score (5090 and 5620, two library entries
for the same measure) in 12 and 11, Number of Cash Transfer Recipients (5239) in 11. Only 37 indicators appear in
4 or more documents, and 68 of the 146 used indicators appear in a single document.

### Never-used IndiKit indicators

205 indicators were matched by no document. The never-used set is concentrated in **detailed WASH outcome measures**
(diarrhoea treatment, water-point queuing time, water storage, latrine maintenance) and **market and income
measures** (selling points, products sold, household expenditure). Only 107 of the 205 are on the website today;
the other 98 are export-only or were removed from the site in November, so proposal writers cannot find them anyway. For 199 of the 205 the pipeline can name the document indicator that comes
closest in wording (Streamlit, Coverage tab), and 89 of them have a document indicator at cosine similarity 0.60 or
higher, which suggests the concept is often present under different wording.

## 5. Novel indicators and clusters

- **874 document indicators have no IndiKit equivalent**, grouped into **605 clusters** by wording similarity
  (greedy clustering at cosine 0.80). 500 clusters are singletons; 105 recur in more than one document; the largest
  has 18 members.
- **The biggest clusters are an artefact of the duplicated proposal.** Clusters 13 to 24, all spanning "six documents
  in six countries", come entirely from the six copies of the We-are-able proposal. Their themes (advocacy initiatives
  by civil society organisations, lobby-and-advocacy capacity, social inclusion of women and girls, local ownership
  and representation of marginalised groups) are real gaps in IndiKit, but they are one proposal, not six.
- After discounting the duplicates, **only 42 clusters recur in two or more distinct documents**, and most of those
  are two versions of the same proposal within one country. The few genuinely cross-document themes are the joint
  response indicators shared by Syria and Yemen (people provided with resources to rebuild livelihood assets, children
  receiving support specific to their needs) and the Burundi returnee indicators on dignity, housing and
  reintegration. Together with the We-are-able advocacy and inclusion themes, these are the candidates to add to
  IndiKit; the rest of the novel set is project-specific output counts.
- **Levels are mostly unknown.** 829 of 1 236 document indicators have level "unclear", because logframe tables
  rarely label rows by output, outcome or impact in a way the parser can read.

## 6. How the pipeline behaved

- **Code did most of the extraction.** 1 118 of 1 236 indicators came from logframe tables parsed in code
  (`table_code`); the LLM extracted only 32 from tables and 26 from text. The novel pass needed just 52 LLM calls.
- **The disclosure step found 345 present pairs out of 6 996 asked** (4.9 %). The retrieval gate at the 90th
  percentile with 60 candidates per document is generous; most asked pairs are correctly answered "not present".
- **The reverse check pays off.** It re-examined 967 unmatched document indicators against their closest IndiKit
  candidates and recovered 93 matches (9.6 %), about a third of all close matches.
- **Narrative-only proposals yield nothing.** 38 queried documents ended with zero indicators although 2 280 pairs
  were asked for them. They are narrative proposals, concept notes and application forms without a logframe in the
  file (the logframe is typically a separate annex). Querying them cost roughly a third of the pair budget for no
  result.
- **The LLM is not fully deterministic.** The six identical copies of the We-are-able proposal returned 38 or 39
  indicators and 33 or 34 unmatched, so counts can differ by one or two between runs of the same document.
- **Few errors.** 7 of 8 086 calls failed, all in the extraction model (gpt-6-luna) returning no JSON for large
  logframe tables; the answer model (gpt-6.1-terra) had no failures. No pair was dropped as unsupported.
- **The cross-encoder re-ranker is unusable on this machine.** On CPU it peaked at 18 to 48 GB of memory and took
  about 10 s per pair even after limiting batch and passage size. Vector-only retrieval (`--no-rerank`) runs at
  about 0.1 s per pair, so the full run used it; the effect on accuracy is unmeasured.

## 7. Cost and runtime

| item | value |
|---|---|
| LLM calls | 8 086 (8 034 answer, 52 extraction) |
| prompt tokens | 22.0 M |
| completion tokens | 0.5 M |
| cost | $24.25 (cap $25.00) |
| cost per document | $0.22 |
| cost per pair | $0.003 |
| summed call latency | 6.5 h |

- The run was started on 2026-11-15 and resumed several times; the resumable design and the answer cache held.
- The earlier estimate with gpt-4o-mini was $3.14 for the same run; the GPT-6 answer model costs about eight times
  more per pair. The default call cap of 1 500 had to be raised to 9 000 for the full run.

## 8. Comparison with the earlier approach

The previous notebook pipeline (fuzzy matching, `outputs/insights_summary.txt`) analysed 75 proposals, extracted 988
indicators and reported 20.6 % present, 16.0 % adapted and 61.3 % missing. The new pipeline covers 115 documents,
finds 1 236 indicators, and reports 29.3 % matched with evidence quotes, reasons and a review trail per row. The two
are not directly comparable (different document sets and match definitions), but the direction is the same: **most
proposal indicators are written outside IndiKit**, and the IndiKit measures that are used are a small, stable core.

## 9. Limitations

- **No measured accuracy.** The gold set (`pipeline/eval/gold/index.csv`) has no verified entries, so none of the
  seven planned experiments (plain-RAG benchmark, ablations, model comparison, gate sensitivity, reverse-check value)
  has run on real labels. Precision and recall of the 362 matches are unknown. The 11 hand-analysed documents of
  10 November 2026 are the starting point.
- **Gate scores are vector-only**; the re-ranked scores were never computed for the full portfolio.
- **Duplicates and cross-references inflate counts.** Six copies of one proposal and 33 documents whose indicators
  also live in an annex mean portfolio totals overstate the number of distinct indicators.
- **Logframes in separate annexes** are only counted when the annex itself is in the proposals folder.
- **Eight scanned PDFs** are only partially parsed; three French documents rely on the multilingual embedding model.
- **Untuned parameters**: the novel-cluster threshold (0.80), the gate percentile (p90) and top-M (60), the
  suggestion floor for closest mentions.
- **Non-determinism** of one to two indicators per document between runs.

## 10. Recommended next steps

1. **Verify a gold set** for 8 to 11 documents from `outputs/latest/review.xlsx` (`export-review`, reviewer
   decisions, `import-review --export-gold`) and run `experiments --split dev`. This is the only way to turn the
   numbers above into measured accuracy and to tune the gate.
2. **De-duplicate the portfolio** before reporting: drop the five extra We-are-able copies and decide how to count
   documents that point to an annex.
3. **Skip narrative-only documents** or gate them harder; 38 documents consumed a third of the pair budget for no
   indicator.
4. **Feed the recurring novel clusters to the IndiKit owners** as candidate indicators, starting with lobby and
   advocacy, civil-society capacity and social inclusion.
5. **Reconsider the never-used tail** of the library: 107 live indicators are not used by any of 115 proposals, and
   six live sub-sectors have no indicator in use.
6. **Ask the IndiKit owners about the November removals.** 46 indicators that proposals use, including the most-used
   one, disappeared from the site; either restore them or tell the country teams what replaces them.

## Sources

- `outputs/latest/portfolio_summary.md`, `run_report.md`, `document_summary.csv`, `document_indicators.csv`,
  `novel_indicators.csv`, `portfolio_*.csv` (run `run-20261115-233434`)
- `docs/phase_1_report.md`, `docs/phase_3_report.md`
- `outputs/kb/preprocess_report.md`, `outputs/insights_summary.txt` (earlier approach)
