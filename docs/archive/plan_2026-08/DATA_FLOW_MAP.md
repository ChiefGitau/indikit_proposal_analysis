# Data Flow & Integration Map: Phases 1–5

## Source-to-Output Data Pipeline

```
COMPLETED WORK (Phases 1–3)
═══════════════════════════

ZOA Export (xlsx)                    Peer Libraries (JSON/CSV)           Proposal Documents (PDF/DOCX)
      ↓                                     ↓                                  ↓
      │                                     │                                  │
      └─────────────────────┬───────────────┴──────────────┬──────────────────┘
                           │                              │
                    Phase 1–3 Notebooks:            Extraction Pipeline:
                    • data_analysis.ipynb           • extraction_utils.py
                    • base_comparison.ipynb         • proposal_indicator_analysis.ipynb
                    • zoa_comparison.ipynb
                           │                              │
                           ↓                              ↓
            ┌─────────────────────────────┐    ┌─────────────────────────────┐
            │   PHASE 1–3 OUTPUTS         │    │   PROPOSAL INDICATORS       │
            ├─────────────────────────────┤    ├─────────────────────────────┤
            │ Comparison Results:         │    │ indicators_extracted.csv    │
            │ • zoa_matches.csv           │    │ • indicator_text            │
            │ • coverage_matrix.csv       │    │ • level                     │
            │ • zoa_unique.csv            │    │ • baseline / target / MoV   │
            │ • semantic_matches.csv      │    │ • sector / doc_id           │
            │ • gaps_base.csv             │    │                             │
            │                             │    │ indicator_matches.csv       │
            │ Sector Benchmarks:          │    │ (existing matches vs        │
            │ • Taxonomy CSVs             │    │  reference library)         │
            │ • Coverage heatmaps         │    │                             │
            │ • Duplicates / near-misses  │    └─────────────────────────────┘
            │                             │
            │ Streamlit App:              │
            │ • zoa_indikit_app/          │
            └─────────────────────────────┘
                           │
                           └─── REFERENCE DATA FOR PHASE 4 ───┐
                                                               │
PHASE 4: PROPOSAL LINKING (WEEKS 1–3)                        │
══════════════════════════════════════════════════════════════════
                                                               │
                                                               ↓
                    ┌─────────────────────────────┐
                    │  PROPOSAL INDICATORS        │
                    │  + ZOA REFERENCE (Phase 1)  │
                    └──────────────┬──────────────┘
                                   │
                        Match to ZOA Reference:
                        ① Exact slug → ZOA ID
                        ② Normalized title → ZOA ID
                        ③ Embedding similarity ≥0.80
                                   │
                                   ↓
            ┌──────────────────────────────────────┐
            │     PHASE 4 OUTPUTS                  │
            ├──────────────────────────────────────┤
            │ proposal_zoa_matches.csv             │
            │ • proposal_id, extracted_indicator   │
            │ • zoa_id, zoa_title                  │
            │ • match_tier, match_score, status    │
            │                                      │
            │ proposal_alignment_scorecard.csv     │
            │ • alignment_score                    │
            │ • quality_score                      │
            │ • sector_coverage                    │
            │ • recommendations                    │
            │                                      │
            │ proposal_alignment_gaps.csv          │
            │ • gap_type, issue                    │
            │ • recommended_zoa_indicator          │
            │                                      │
            │ proposal_indicator_linking_guide.md  │
            │ • Decision flowchart                 │
            │ • Examples & best practices          │
            │                                      │
            │ proposal_me_template.xlsx            │
            │ • Sector dropdown                    │
            │ • Pre-populated ZOA indicators       │
            │ • Quality checklist                  │
            └─────────────────┬────────────────────┘
                              │
                              └─── LINKAGE + GUIDANCE FOR PHASE 5 ───┐
                                                                      │
PHASE 5: REPORTING & ROADMAP (WEEKS 4–5)                            │
══════════════════════════════════════════════════════════════════════
                                                                      │
                                                                      ↓
            ┌──────────────────────────────────────┐
            │   CONSOLIDATE ALL PHASES (1–4)       │
            │   + Generate Reports & Dashboard     │
            └──────────────────┬───────────────────┘
                               │
                               ├─ Executive Summary
                               ├─ Sector Profiles
                               ├─ ZOA-Unique Assessment
                               ├─ Proposal Alignment Patterns
                               └─ Dashboard Metrics
                               │
                               ↓
            ┌──────────────────────────────────────┐
            │     PHASE 5 OUTPUTS                  │
            ├──────────────────────────────────────┤
            │ benchmark_consolidated.csv           │
            │ • sector, adoption_distribution      │
            │ • coverage_gaps, unique_count        │
            │                                      │
            │ benchmark_executive_summary.md       │
            │ • ZOA footprint & adoption profile   │
            │ • Top 3–5 recommendations            │
            │                                      │
            │ sector_profiles_detailed.md          │
            │ • Per-sector KPIs by level           │
            │ • Gaps & unique indicators           │
            │                                      │
            │ zoa_unique_strategic_assessment.md   │
            │ • Risk matrix                        │
            │ • Proposed actions                   │
            │                                      │
            │ dashboard_metrics.csv                │
            │ • Adoption distribution (histogram)  │
            │ • Sector heatmap data                │
            │ • Proposal scatter data              │
            │ • Risk matrix data                   │
            │                                      │
            │ sample_visualizations/               │
            │ • PNG charts for reports             │
            │                                      │
            │ implementation_roadmap.md            │
            │ • Phase A–D timeline (12 months)     │
            │ • Success metrics & KPIs             │
            │ • Resource plan                      │
            │                                      │
            │ change_management_plan.md            │
            │ • Kickoff + training schedule        │
            │ • Feedback loops                     │
            └──────────────────┬───────────────────┘
                               │
                               ↓
                    ╔════════════════════════╗
                    ║   READY FOR ROLLOUT    ║
                    ║  Q4 2026 & Beyond      ║
                    ╚════════════════════════╝
                               │
                    Pilot Projects (3–5) using:
                    • proposal_me_template.xlsx
                    • linking_guide.md
                    • Approval workflows
                               │
                    ← Monthly feedback → Refinement ← Quarterly reviews
```

---

## Key Integration Points

### Phase 4 → Phase 5 Dependency
```
proposal_zoa_matches.csv
    ↓ (analyze patterns)
proposal_alignment_scorecard.csv
    ↓ (aggregate by sector/proposal)
proposal_alignment_gaps.csv
    ↓ (identify most common gaps)
→ Input to: benchmark_consolidated.csv
           sector_profiles_detailed.md
           implementation_roadmap.md
```

### Phase 1–3 → Phase 4 Inputs
```
zoa_matches.csv (adoption counts)
    ↓ (enrich proposal matches with adoption band)
proposal_zoa_matches.csv
    ↓
(Flag ZOA-unique indicators used in proposals)
    ↓
proposal_alignment_scorecard.csv
    ↓
(Risk profile from Phase 3)
→ Output: Recommendations for proposal developers
```

### Tools & Notebooks

```
Notebook Workflow:
═════════════════

Phase 1–3 (Complete):
  data_analysis.ipynb → raw indicators (JSON/CSV)
  base_comparison.ipynb → adoption vs base
  zoa_comparison.ipynb → adoption vs 9 peers
  
Phase 4 (New):
  proposal_indicator_analysis_linked.ipynb
    ├─ Load: proposal indicators + Phase 1–3 data
    ├─ Matching: exact slug → title → embedding
    ├─ Output: proposal_zoa_matches.csv + scorecard
    └─ Visualize: alignment scatter, quality heatmap
  
Phase 5 (New):
  benchmark_reporting.ipynb
    ├─ Consolidate: all Phase outputs
    ├─ Reports: executive summary + sector profiles
    ├─ Assessment: ZOA-unique strategic review
    ├─ Dashboard: prepare CSV + sample charts
    └─ Roadmap: generate implementation plan
```

---

## Data File Locations

**Input Data (Phases 1–3):**
```
MEAL_Proposal-main/MEAL_Proposal-main/
├─ data/indicators/
│  ├─ Export indikit 03022026.xlsx          [ZOA reference]
│  ├─ Export indikit.csv
│  └─ data_analysis/
│     ├─ Taxonomy/indicators_per_*.csv
│     └─ Coverage/coverage_sector_level.csv
│
├─ notebooks/
│  ├─ data_analysis.ipynb
│  ├─ base_comparison.ipynb
│  ├─ zoa_comparison.ipynb
│  ├─ data/{domain}/indicators.{json,csv}   [10 organizations]
│  └─ comparison_zoa/                        [Comparison outputs]
│     ├─ zoa_matches.csv
│     ├─ coverage_matrix.csv
│     ├─ zoa_unique.csv
│     └─ ... (near-misses, gaps, semantic)
│
└─ zoa_indikit_app/                         [Streamlit app]
   └─ app.py + data/
```

**Proposal Data (Extracted):**
```
proposal_investigated/
├─ proposals/                                [PDF/DOCX source]
├─ outputs/
│  ├─ indicators_extracted.csv               [LLM extracted indicators]
│  ├─ indicator_matches.csv                  [Existing matches]
│  └─ figures/                               [Visualizations]
└─ proposal_indicator_analysis.ipynb         [Extraction pipeline]
```

**Phase 4–5 Outputs:**
```
proposal_investigated/
├─ outputs/                                  [Phase 4–5 deliverables]
│  ├─ proposal_zoa_matches.csv
│  ├─ proposal_alignment_scorecard.csv
│  ├─ proposal_alignment_gaps.csv
│  ├─ proposal_me_template.xlsx
│  ├─ benchmark_consolidated.csv
│  ├─ dashboard_metrics.csv
│  ├─ sample_visualizations/
│  └─ reports/                               [Markdown reports]
│     ├─ proposal_indicator_linking_guide.md
│     ├─ proposal_me_template_guide.md
│     ├─ benchmark_executive_summary.md
│     ├─ sector_profiles_detailed.md
│     ├─ zoa_unique_strategic_assessment.md
│     ├─ implementation_roadmap.md
│     └─ change_management_plan.md
│
└─ plan/                                     [Planning docs]
   ├─ indicator_benchmark_zoa_linking_plan.md [Full plan, 499 lines]
   └─ PHASE_4_5_SUMMARY.md                   [Quick reference]
```

---

## Quick Lookup: Data → Output

| Need... | Source | Location |
|---------|--------|----------|
| ZOA reference (303 indicators) | Phase 1 | `Export indikit 03022026.xlsx` |
| Adoption counts (per ZOA indicator) | Phase 1–3 | `notebooks/comparison_zoa/zoa_matches.csv` |
| Sector benchmarks | Phase 2 | `data/indicators/data_analysis/Taxonomy/` |
| ZOA-unique indicators + rationale | Phase 3 | `notebooks/comparison_zoa/zoa_unique.csv` |
| Proposal indicators to match | Phase 4 input | `outputs/indicators_extracted.csv` |
| Proposal alignment scores | Phase 4 | `outputs/proposal_alignment_scorecard.csv` |
| Linking decision guide | Phase 4 | `outputs/reports/proposal_indicator_linking_guide.md` |
| M&E template | Phase 4 | `outputs/proposal_me_template.xlsx` |
| Executive summary | Phase 5 | `outputs/reports/benchmark_executive_summary.md` |
| Implementation roadmap | Phase 5 | `outputs/reports/implementation_roadmap.md` |

---

**Status:** Ready to execute Phases 4–5 🚀 | Timeline: Aug 13 – Sep 17, 2026
