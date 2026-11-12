# Updated Plan Summary: Indicator Benchmark & ZOA-Only Linking

**Date:** August 13, 2026 | **Status:** ✅ Phases 1–3 Complete | 🚀 Phases 4–5 Ready

---

## What's Already Done (Phases 1–3) ✅

### Phase 1: ZOA Indicator Universe
- 303 ZOA indicators classified across 30 sectors
- Adoption scored against 9 peer organizations (base, ADRA, IRW, Tearfund, WHH, Winrock, TDH, Helvetas, Save the Children)
- Results: `notebooks/comparison_zoa/zoa_matches.csv`, `coverage_matrix.csv`, `zoa_unique.csv`

### Phase 2: Sector Benchmarks
- Sector-level coverage analysis (Impact/Outcome/Output mix)
- Gap analysis: peer indicators ZOA lacks
- Duplicate/near-duplicate detection across sectors
- Results: `data_analysis/Taxonomy/`, `Coverage/`, `indikit_analysis_log/Duplicates/`

### Phase 3: ZOA-Unique Indicators
- 42+ ZOA-only indicators identified (adoption = 0)
- Strategic uniqueness contexts (Tearfund, WHH, Save the Children deep dives)
- Near-miss analysis (similarity 0.60–0.80)
- Results: `zoa_unique.csv`, `semantic_matches.csv`, `*_near_misses.csv`

### Tools & Infrastructure
- **Streamlit App** (`zoa_indikit_app/`) — Interactive profiler with taxonomy, coverage, semantic maps, duplicates detection
- **Matching Pipeline** — Proven ladder: exact slug → normalized title → embedding (cosine ≥ 0.80)

---

## What's Next (Phases 4–5) 🚀

### Phase 4: Proposal-to-ZOA Linking (2–3 weeks, Aug 13 – Sep 3)

**Goal:** Connect proposal indicators to ZOA reference; create reusable linking framework.

**Deliverables:**
- `proposal_zoa_matches.csv` — All proposals matched to ZOA reference (match tier + confidence)
- `proposal_alignment_scorecard.csv` — Per-proposal scores: alignment, quality, sector coverage
- `proposal_alignment_gaps.csv` — Gaps by proposal/sector
- `proposal_indicator_linking_guide.md` — Decision flowchart for developers
- `proposal_me_template.xlsx` — Ready-to-use sector-aware template

**Activities:**
1. Build matching pipeline (reuse `zoa_comparison.ipynb` code)
2. Score first 5 pilot proposals
3. Draft linking decision guide (flowchart + examples)
4. Create Excel template with sector dropdown + auto-population
5. Review with M&E team; collect feedback

**Success:** 80%+ matching accuracy on pilots; M&E team confident in workflow

---

### Phase 5: Reporting & Implementation (2 weeks, Sep 3 – Sep 17)

**Goal:** Package findings into reports and action plan for leadership.

**Deliverables:**
- `benchmark_executive_summary.md` — Key findings + top 3–5 recommendations
- `sector_profiles_detailed.md` — Per-sector KPIs, gaps, unique indicators
- `zoa_unique_strategic_assessment.md` — Risk matrix + recommended actions (standardize/sunsetting)
- `dashboard_metrics.csv` + sample charts — For interactive dashboard
- `implementation_roadmap.md` — Phased 12-month rollout + success metrics
- `change_management_plan.md` — Kickoff, training, feedback loops

**Activities:**
1. Consolidate all Phase 1–4 outputs into reports
2. Generate visualizations (adoption distribution, sector heatmap, risk matrix, proposal scatter)
3. Draft implementation roadmap + resource plan
4. Schedule team training + leadership sign-off
5. Prepare kickoff presentation

**Success:** Reports endorsed by leadership; team trained; rollout approved

---

## Timeline at a Glance

```
Aug 13 ────────────────────────── Sep 3 ────────────────────────── Sep 17
       ← PHASE 4 (2–3 weeks) →       ← PHASE 5 (2 weeks) →
  
Week 1: Matching pipeline coded       Week 4: Reports drafted
Week 2: Proposal scores ready        Week 5: Dashboard + roadmap finalized
Week 3: Linking guide + template         ✓ Rollout kickoff scheduled
```

---

## Key Data Sources (All Available)

| Source | Location | Use |
|--------|----------|-----|
| ZOA reference | `Export indikit 03022026.xlsx` | Matching baseline |
| Proposal indicators | `proposal_indicator_analysis.ipynb` outputs | Linking input |
| Adoption scores | `notebooks/comparison_zoa/zoa_matches.csv` | Phase 4 enrichment |
| Sector benchmarks | `data/indicators/data_analysis/` | Phase 5 reporting |
| Peer context | `notebooks/data/{domain}/indicators.*` | Gap analysis |

---

## Immediate Actions (This Week)

- [ ] Create `proposal_indicator_analysis_linked.ipynb` 
- [ ] Load proposal indicators + ZOA reference
- [ ] Implement matching ladder from `zoa_comparison.ipynb`
- [ ] Test on 3 pilot proposals
- [ ] Identify common patterns + quick wins

---

## Questions This Will Answer

**By Phase 4 completion:**
- How aligned are ZOA proposals to the ZOA reference library?
- What are the most common gaps (indicators proposals use that ZOA lacks)?
- Are proposals using ZOA-unique indicators appropriately?
- What guidance do proposal developers need?

**By Phase 5 completion:**
- Should ZOA adopt any new indicators from peers?
- Which ZOA-unique indicators should be reviewed or sunsetted?
- How does ZOA's library compare to peer organizations?
- What's the implementation plan for improving proposal indicator quality?

---

## Success Criteria

**Phase 4:**
- ✅ Matching pipeline achieves ≥80% accuracy (manual validation)
- ✅ M&E template adopted by all sector leads
- ✅ Linking guide clarity score ≥4/5 from M&E team
- ✅ First 5 proposals scored; gaps identified

**Phase 5:**
- ✅ Benchmark reports accepted by leadership
- ✅ Rollout roadmap feasibility confirmed
- ✅ Pilot proposals (3–5) using template with >70% alignment
- ✅ Team trained; M&E staff confident

**Year 1:**
- ✅ 80%+ of new proposals use template (by Q2)
- ✅ Average alignment score ≥70% (by Q3)
- ✅ 90%+ indicators have baseline/target/MoV (by Q3)
- ✅ Zero proposals approved with unreviewed ZOA-unique indicators

---

## Document Location

📄 **Full Plan:** `plan/indicator_benchmark_zoa_linking_plan.md` (499 lines, comprehensive)

**File Structure:**
- Sections 1–2: Completed work + remaining work
- Sections 3–4: Technical workflow, pipeline, data sources
- Sections 5–8: Timeline, success criteria, deliverables, immediate actions
- Sections 9–11: Dependencies, glossary, version history

---

## Who's Involved

| Role | Responsibilities |
|------|-----------------|
| **Data Analyst** | Matching pipeline, proposal scoring, reports, dashboard |
| **M&E Lead** | Linking guide review, template design, team training, approvals |
| **Sector Leads** | Validate ZOA-unique rationale, approve new indicators |
| **M&E Team** | Pilot proposals, feedback, adoption tracking |
| **Leadership** | Sign-off on roadmap, resource allocation, rollout approval |

---

**Ready to start Phase 4? Let's build the linking pipeline! 🚀**
