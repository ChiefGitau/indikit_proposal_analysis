# Indicator Benchmark & ZOA-Only Linking Analysis Plan

**Objective:** Integrate comprehensive ZOA indicator benchmarking (already completed in MEAL_Proposal-main) with proposal indicator linking and validation, enabling best-practice proposal M&E design and ZOA library optimization.

**Status:** Phases 1–3 complete; Phases 4–5 in focus.

---

## 1. Completed Work Summary (Phases 1–3)

### 1.1 Phase 1: ZOA Indicator Universe — ✅ COMPLETE

**Deliverables:**
- **ZOA Reference Library:** 303 indicators across 30 sectors, with full taxonomy
  - Source: `Export indikit 03022026.xlsx` + `Export indikit.csv`
  - Structure: Section → Sector → Sub-Sector → Level → Indicator Name + Wording (ENG/FRA)
- **Adoption Classification:**
  - ZOA indicators matched against 9 peer organizations (base + ADRA, IRW, Tearfund, WHH, Winrock, TDH, Helvetas, Save the Children)
  - Adoption counts per indicator: 0 (unique) to 9 (universal)
  - 10 indicators adopted by all peers → ultra-standard
  - ~216 indicators shared with base → sector-standard candidates

**Key Data:**
- `notebooks/comparison_zoa/zoa_matches.csv` — full match ladder (ZOA × site, match tier, similarity)
- `notebooks/comparison_zoa/coverage_matrix.csv` — 303 ZOA indicators × 9 sites coverage
- `notebooks/comparison_zoa/zoa_unique.csv` — 42+ ZOA-only indicators (adoption = 0)
- `notebooks/comparison_zoa/summary.csv` — per-site adoption statistics

### 1.2 Phase 2: Sector Benchmarks — ✅ COMPLETE

**Deliverables:**
- **Sector Coverage Analysis:**
  - Per-sector KPIs: count by level, adoption distribution, coverage gaps
  - Taxonomy available: `data/indicators/data_analysis/Taxonomy/indicators_per_sector.csv`
  - Coverage heatmaps: sector × level, sector × sub-sector
- **Gap Analysis:**
  - Indicators base/peers measure that ZOA lacks: `notebooks/comparison_zoa/gaps_base.csv`
  - Sector × adoption heatmap shows where ZOA is weak vs peers
- **Data Quality Artifacts:**
  - Missing values, wording length distribution, language coverage
  - Duplicate/near-duplicate pairs: `data/indicators/data_analysis/indikit_analysis_log/Duplicates/`

**Key Data:**
- `data/indicators/data_analysis/coverage/coverage_sector_level.csv`
- `data/indicators/data_analysis/indikit_analysis_log/Taxonomy/indicators_per_*.csv`
- `data/indicators/data_analysis/indikit_analysis_log/Coverage/` — sector mixes + heatmaps

### 1.3 Phase 3: ZOA-Unique Indicator Profile — ✅ COMPLETE (Partial)

**Deliverables:**
- **ZOA-Unique Inventory:**
  - 42+ indicators with adoption = 0 (no peer carries them)
  - Identified in: `notebooks/comparison_zoa/zoa_unique.csv`
- **Uniqueness Context:**
  - Deep dives for Tearfund, WHH, Save the Children (custom platforms)
  - Near-miss tables: indicators close to ZOA's (similarity 0.60–0.80): `notebooks/comparison_zoa/*_near_misses.csv`
  - Semantic best-match analysis: `notebooks/comparison_zoa/semantic_matches.csv`

**Key Data:**
- `notebooks/comparison_zoa/zoa_unique.csv`
- `notebooks/comparison_zoa/{tearfund,whh,savethechildren}_near_misses.csv`
- `notebooks/comparison_zoa/semantic_matches.csv`

### 1.4 Tools & Infrastructure — ✅ COMPLETE

**Streamlit Dashboard:**
- `zoa_indikit_app/` — Interactive profiler for ZOA indicators
  - Pages: Taxonomy, Coverage, Text Profile, Semantic Map, Duplicates, Data Quality, Analysis Log, Proposal Overview, Site Comparison
  - Embeddings & semantic analysis ready (optional ML dependencies)
  - Live site comparison scraper (BeautifulSoup)

**Matching & Analysis Pipeline:**
- `notebooks/data_analysis.ipynb` — Scraping all 10 IndiKit sites
- `notebooks/base_comparison.ipynb` — Base vs peers
- `notebooks/zoa_comparison.ipynb` — ZOA as reference (main comparison)
- Matching ladder: exact slug → normalized title → embedding cosine ≥ 0.80

---

## 2. Remaining Work: Phases 4–5

### Phase 4: LLM-Based Indicator Linking (Simplified)

**Goal:** Use LLM to link proposal extracted indicators to ZOA reference library and generate a simple status table.

**Implementation:** Enhanced `proposal_indicator_analysis.ipynb` (add one new cell block)

#### 4.1 Load ZOA Reference + Proposal Indicators

- [ ] Load ZOA reference from Phase 1–3:
  - `notebooks/data/zoa_indikit_net/indicators.csv` or `Export indikit.csv`
  - Key columns: `Indicator Name`, `Indicator Wording - ENG`, `Sector`, `Sub-Sector`, `Indicator Level`
  
- [ ] Load extracted proposal indicators:
  - `outputs/indicators_extracted.csv` (from existing extraction pipeline)
  - Key columns: `indicator_text`, `level`, `sector`, `doc_id`, `baseline`, `target`, `means_of_verification`

#### 4.2 LLM Linking Function

- [ ] **Create LLM linking prompt:**
  - Input: Proposal indicator text + list of ZOA indicators
  - Task: Match proposal indicator to best ZOA indicator OR classify as (Missing/Different/Present)
  - Output: JSON with `zoa_indicator`, `match_status` (Present/Different/Missing), `confidence`, `reason`

```python
def link_indicator_to_zoa(proposal_indicator, zoa_indicators_list, client):
    prompt = f"""
    Proposal Indicator: "{proposal_indicator}"
    
    ZOA Reference Indicators:
    {zoa_indicators_list}
    
    Task: Match the proposal indicator to a ZOA indicator or classify it as:
    - "Present": Found exact/very similar match in ZOA
    - "Different": Found related ZOA indicator but worded differently
    - "Missing": No ZOA indicator covers this
    
    Return JSON: {{"zoa_indicator": "...", "status": "Present/Different/Missing", "confidence": 0-1, "reason": "..."}}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

#### 4.3 Generate Linking Status Table

- [ ] For each proposal, create a status table:

| Indicator | Type | ZOA Match | Sector | Status | Confidence |
|-----------|------|-----------|--------|--------|------------|
| % Households with food security | Outcome | Food Security - Dietary Diversity | Food Security | Present | 0.95 |
| Number of jobs created | Output | Employment - Jobs | Employment | Different | 0.80 |
| Water quality testing | Outcome | (none) | WASH | Missing | 0.0 |

- [ ] **Columns:**
  - `indicator` — extracted proposal indicator text
  - `type` — level (Impact/Outcome/Output)
  - `zoa_match` — matched ZOA indicator (if found)
  - `sector` — proposal sector
  - `status` — Present / Different / Missing
  - `confidence` — LLM confidence (0–1)

- [ ] **Output:** `proposal_indicator_status.csv`
  - Rows: All extracted indicators × proposals
  - Summary stats per proposal: % Present, % Different, % Missing

**Deliverables:**
- Updated `proposal_indicator_analysis.ipynb` with new "Indicator Linking" section
- `proposal_indicator_status.csv` (status table for all proposals)

---

### Phase 5: Simple Summary & Insights (Simplified)

**Goal:** Analyze linking results; provide simple summary table and insights.

#### 5.1 Summary Table by Proposal

- [ ] **Create proposal-level summary:** `proposal_indicator_summary.csv`

| Proposal | Total Indicators | Present | Different | Missing | % Present | % Different | % Missing |
|----------|-----------------|---------|-----------|---------|-----------|-------------|-----------|
| Proposal A | 15 | 12 | 2 | 1 | 80% | 13% | 7% |
| Proposal B | 22 | 16 | 4 | 2 | 73% | 18% | 9% |
| Proposal C | 10 | 7 | 2 | 1 | 70% | 20% | 10% |

- [ ] **Columns:**
  - `proposal` — proposal name/ID
  - `total_indicators` — total extracted indicators
  - `present` — matched to ZOA reference
  - `different` — related ZOA indicator exists but worded differently
  - `missing` — no ZOA indicator covers this
  - Percentage columns for each status

#### 5.2 Sector-Level Insights

- [ ] **Create sector summary:** `sector_indicator_summary.csv`

| Sector | Proposals | Total Indicators | Avg % Present | Avg % Different | Avg % Missing | Gap Areas |
|--------|-----------|-----------------|---------------|-----------------|---------------|-----------|
| WASH | 5 | 65 | 75% | 15% | 10% | Sanitation monitoring |
| Food Security | 8 | 120 | 82% | 12% | 6% | Livelihood diversification |
| Education | 3 | 45 | 68% | 18% | 14% | Learning outcomes measurement |

#### 5.3 Key Insights (Simple)

- [ ] **Generate simple insights:**
  - Average % Present across all proposals (target: >75%)
  - Most common missing indicators (opportunity for ZOA library expansion)
  - Most common "Different" indicators (where proposals adapt ZOA indicators)
  - By sector: which sectors have highest/lowest coverage
  
- [ ] **Output:** `insights_summary.txt`

```
PROPOSAL INDICATOR LINKING SUMMARY
═══════════════════════════════════

Overall Statistics:
  Total proposals analyzed: 12
  Total indicators extracted: 245
  Average % Present: 76%
  Average % Different: 15%
  Average % Missing: 9%

Top Missing Indicators (Across All Proposals):
  1. Market price monitoring (15 proposals)
  2. Community participation index (12 proposals)
  3. Water quality testing (10 proposals)

Most Adapted Indicators (Different status):
  1. Household income (12 proposals adapt ZOA wording)
  2. School attendance (10 proposals)
  
By Sector:
  WASH: 75% present, 15% different, 10% missing
  Food Security: 82% present, 12% different, 6% missing
  Education: 68% present, 18% different, 14% missing
```

**Deliverables:**
- `proposal_indicator_summary.csv` (by-proposal summary)
- `sector_indicator_summary.csv` (by-sector summary)
- `insights_summary.txt` (key findings)

---

## 3. Technical Workflow & Data Pipeline

### 3.1 Phase 4 Implementation (LLM Linking in Notebook)

---

## 3. Technical Implementation (Simplified)

### 3.1 Phase 4: LLM Indicator Linking in Notebook

```python
# Load ZOA reference
zoa_df = pd.read_csv('MEAL_Proposal-main/notebooks/data/zoa_indikit_net/indicators.csv')
zoa_indicators_text = zoa_df[['Indicator Name', 'Indicator Wording - ENG', 'Sector']].to_dict('records')

# LLM linking function
def link_indicator_to_zoa(proposal_indicator, zoa_list, client):
    """Use LLM to match proposal indicator to ZOA reference."""
    zoa_text = "\n".join([f"- {ind['Indicator Name']} ({ind['Sector']})" for ind in zoa_list[:20]])
    
    prompt = f"""
    Given this proposal indicator, determine if it matches a ZOA indicator:
    
    Proposal Indicator: "{proposal_indicator}"
    
    ZOA Reference Indicators:
    {zoa_text}
    ...
    
    Classify as one of:
    - "Present": Exact or very similar match found
    - "Different": Related indicator exists but wording differs significantly
    - "Missing": No ZOA indicator covers this
    
    Return JSON: {{"zoa_indicator": "name or null", "status": "Present|Different|Missing", "confidence": 0.0-1.0}}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# Apply to all indicators
results = []
for idx, row in indicators_df.iterrows():
    result = link_indicator_to_zoa(row['indicator_text'], zoa_indicators_text, client)
    results.append({
        'indicator_text': row['indicator_text'],
        'zoa_match': result.get('zoa_indicator'),
        'match_status': result['status'],
        'confidence': result['confidence'],
        'doc_id': row['doc_id']
    })

linking_df = pd.DataFrame(results)
indicators_df = indicators_df.merge(linking_df, on=['indicator_text', 'doc_id'], how='left')
```

**Outputs:**
- Enhanced `indicators_extracted.csv` with columns: `zoa_match`, `match_status`, `confidence`
- `proposal_indicator_status.csv` — Full status table (rows = all indicators)

### 3.2 Phase 5: Generate Summary Tables

```python
# Summary by proposal
proposal_summary = indicators_df.groupby('doc_id').agg({
    'match_status': 'count',  # total
    'match_status': lambda x: (x == 'Present').sum(),  # present
    'match_status': lambda x: (x == 'Different').sum(),  # different
    'match_status': lambda x: (x == 'Missing').sum()  # missing
}).rename(columns={'match_status': 'total'})

# Add percentages
proposal_summary['pct_present'] = (proposal_summary['present'] / proposal_summary['total'] * 100).round(1)
proposal_summary['pct_different'] = (proposal_summary['different'] / proposal_summary['total'] * 100).round(1)
proposal_summary['pct_missing'] = (proposal_summary['missing'] / proposal_summary['total'] * 100).round(1)

proposal_summary.to_csv(OUTPUTS_DIR / 'proposal_indicator_summary.csv')

# Summary by sector
sector_summary = indicators_df.groupby('sector').agg({
    'match_status': 'count',
    'match_status': lambda x: (x == 'Present').sum(),
    'match_status': lambda x: (x == 'Different').sum(),
    'match_status': lambda x: (x == 'Missing').sum()
})

sector_summary.to_csv(OUTPUTS_DIR / 'sector_indicator_summary.csv')
```

**Outputs:**
- `proposal_indicator_summary.csv`
- `sector_indicator_summary.csv`
- `insights_summary.txt` (key findings)

---

## 4. Timeline & Milestones

| Phase | Duration | Effort | Key Milestones |
|-------|----------|--------|----------------|
| Phase 1 | 2026-06 to 2026-07 | ✅ Complete | ZOA universe defined; adoption bands assigned |
| Phase 2 | 2026-07 to 2026-08 | ✅ Complete | Sector scores + gap analysis done; near-duplicates identified |
| Phase 3 | 2026-07 to 2026-08 | ✅ Complete | ZOA-unique inventory; strategic rationale documented |
| **Phase 4** | **2026-08 to 2026-09** | **2–3 weeks** | **Linking pipeline built; template ready; first 3 proposals scored** |
| **Phase 5** | **2026-09 to 2026-10** | **2 weeks** | **Reports + roadmap finalized; team trained; rollout plan signed off** |
| **Integration** | **2026-10 onward** | **Ongoing** | **Pilot deployment; feedback loops; quarterly refreshes** |

### Phase 4 Breakdown (2–3 weeks)

| Week | Deliverable | Lead | Status |
|------|-------------|------|--------|
| 1 | Matching pipeline coded + tested on 5 pilot proposals | Analyst | |
| 1 | proposal_zoa_matches.csv + proposal_alignment_scorecard.csv | Analyst | |
| 2 | Linking decision guide + flowchart finalized | M&E + Analyst | |
| 2 | M&E template (Excel) built + documented | M&E | |
| 3 | Linking guide review + pilot proposal feedback | M&E team | |
| 3 | Phase 4 summary report (patterns, quick wins, gaps) | Analyst | |

### Phase 5 Breakdown (2 weeks)

| Week | Deliverable | Lead | Status |
|---|---|---|---|
| 1 | Benchmark reports drafted (executive + sector profiles) | Analyst | |
| 1 | ZOA-unique strategic assessment finalized | Analyst + Strategy | |
| 2 | Dashboard data + visualizations ready | Analyst | |
| 2 | Implementation roadmap + change management plan | M&E + Leadership | |
| 2 | Rollout kickoff + team training scheduled | M&E | |

---

## 5. Success Criteria & KPIs

### Phase 4 Success (Linking Framework)

- ✅ Matching pipeline achieves ≥80% accuracy on pilot proposals (manual validation)
- ✅ M&E template adopted by all sector leads (training completion)
- ✅ Average proposal alignment score on pilot = baseline (shows pipeline works)
- ✅ Linking guide clarity score ≥4/5 from M&E team (survey)
- ✅ First 5 proposals re-scored using new pipeline; gaps identified

### Phase 5 Success (Reporting & Roadmap)

- ✅ Benchmark reports accepted by leadership (sign-off)
- ✅ Implementation roadmap feasibility confirmed (resource review)
- ✅ Dashboard live and refreshing quarterly (data checks)
- ✅ Pilot rollout (3–5 proposals using template + linking guide) with >70% alignment
- ✅ Team training completed; M&E staff confident in workflow (post-training survey)

### Long-Term Success (Year 1)

- ✅ 80%+ of all new proposals use template by Q2
- ✅ Average proposal alignment score ≥70% by Q3
- ✅ Quality metrics: 90%+ indicators have baseline/target/MoV by Q3
- ✅ Zero proposals approved with unreviewed ZOA-unique indicators (process control)
- ✅ Sector lead feedback incorporated in Q4 benchmark refresh
- ✅ Dashboard used by leadership for portfolio-level MEAL decisions (adoption indicator)

---

## 6. Key Analytical Questions & Answers

By completion of Phases 4–5, the framework will answer:

### Phase 4: Proposal Linking

1. **How aligned are ZOA proposals to the ZOA reference library?**
   - Answer: `proposal_alignment_scorecard.csv` + average alignment score trend

2. **What are the common gaps (indicators proposals use that ZOA lacks)?**
   - Answer: `proposal_alignment_gaps.csv` + gap frequency by sector

3. **Are proposals using ZOA-unique indicators appropriately?**
   - Answer: Proposals with >30% ZOA-unique adoption flagged for review + rationale check

4. **What guidance do proposal developers need?**
   - Answer: `proposal_indicator_linking_guide.md` + sector-specific recommendations

### Phase 5: Strategy & Roadmap

5. **Should ZOA adopt any new indicators from peer libraries?**
   - Answer: `benchmark_consolidated.csv` top gaps ranked by adoption count + strategic fit

6. **Which ZOA-unique indicators should be reviewed or sunsetted?**
   - Answer: `zoa_unique_strategic_assessment.md` risk matrix + recommendations

7. **How does ZOA's library compare to peer organizations?**
   - Answer: Benchmark reports + per-sector comparison charts

8. **What's the implementation plan for improving proposal indicator quality?**
   - Answer: `implementation_roadmap.md` with phased approach + success metrics

---

## 7. Outputs & Deliverables Checklist

```
proposal_indicator_analysis.ipynb/ (ENHANCED with Phases 4–5)
│
├─ PHASE 4: Proposal-to-ZOA Linking
│  ├─ proposal_zoa_matches.csv
│  │   └─ columns: proposal_id, extracted_indicator, zoa_id, zoa_title, match_tier, match_score, match_status
│  ├─ proposal_alignment_scorecard.csv
│  │   └─ columns: proposal_id, total_indicators, zoa_matched, alignment_score, quality_score, sector_coverage
│  ├─ proposal_alignment_gaps.csv
│  │   └─ columns: proposal_id, gap_type, issue, recommended_zoa_indicator
│  ├─ proposal_indicator_linking_guide.md
│  │   ├─ Decision flowchart
│  │   ├─ Real proposal examples → ZOA matches
│  │   ├─ Risk assessment template
│  │   └─ Approval workflow
│  ├─ proposal_me_template.xlsx
│  │   ├─ Sector-aware pre-population
│  │   ├─ Quality checklist
│  │   └─ New indicator approval tracker
│  └─ proposal_me_template_guide.md
│
├─ PHASE 5: Reporting & Dashboard
│  ├─ benchmark_consolidated.csv
│  │   └─ sector, zoa_count, adoption_distribution, gaps, unique_profile, recommendations
│  ├─ benchmark_executive_summary.md
│  │   ├─ ZOA footprint & adoption profile
│  │   ├─ Sector strengths/weaknesses
│  │   ├─ ZOA-unique indicators summary
│  │   └─ Top 3–5 priority recommendations
│  ├─ sector_profiles_detailed.md (or per-sector pages)
│  │   ├─ Per-sector KPIs by level
│  │   ├─ Adoption distribution + gaps
│  │   ├─ Near-misses & ZOA-unique within sector
│  │   └─ Recommendations
│  ├─ zoa_unique_strategic_assessment.md
│  │   ├─ Uniqueness taxonomy & count per category
│  │   ├─ Risk matrix (comparability vs resource)
│  │   ├─ Strategic rationale by category
│  │   └─ Proposed actions (standardize / sunsetting / keep)
│  ├─ dashboard_metrics.csv
│  │   ├─ Adoption distribution (histogram data)
│  │   ├─ Sector adoption band heatmap (data)
│  │   ├─ Coverage depth by sector (stacked bar data)
│  │   ├─ Proposal alignment scatter (data + colors)
│  │   └─ ZOA-unique risk matrix (data)
│  ├─ sample_visualizations/
│  │   ├─ adoption_distribution.png
│  │   ├─ sector_adoption_heatmap.png
│  │   ├─ proposal_alignment_scatter.png
│  │   └─ zoa_unique_risk_matrix.png
│  ├─ implementation_roadmap.md
│  │   ├─ Phase A–D timeline (Months 1–12)
│  │   ├─ Success metrics + KPIs
│  │   ├─ Resource plan (roles, capacity, training)
│  │   └─ Risk mitigations
│  └─ change_management_plan.md
│      ├─ Stakeholder map (MEAL, leadership, sectors, pilots)
│      ├─ Communication plan (kickoff, monthly updates, feedback cycles)
│      ├─ Training schedule & materials
│      └─ FAQ + escalation process
│
└─ Supporting Notebooks
   ├─ proposal_indicator_analysis_linked.ipynb (Phase 4)
   │  ├─ Load proposal_indicator_analysis outputs
   │  ├─ Match to ZOA reference using matching ladder
   │  └─ Generate proposal scorecards + visualizations
   └─ benchmark_reporting.ipynb (Phase 5)
      ├─ Consolidate all Phase 1–4 outputs
      ├─ Generate reports + dashboard data
      └─ Render charts
```

---

## 8. Next Steps (Immediate Actions)

### For Phase 4 (Weeks 1–3, Aug 13 – Sep 3)

- [ ] **Week 1:** 
  - [ ] Create `proposal_indicator_analysis_linked.ipynb` (skeleton + imports)
  - [ ] Load existing proposal indicators + ZOA reference
  - [ ] Implement matching ladder (reuse `zoa_comparison.ipynb` code)
  - [ ] Test on 3 pilot proposals
  
- [ ] **Week 2:**
  - [ ] Generate proposal_zoa_matches.csv + proposal_alignment_scorecard.csv
  - [ ] Identify patterns (common gaps, alignment score distribution)
  - [ ] Draft linking decision guide (flowchart + examples)
  
- [ ] **Week 3:**
  - [ ] Build Excel M&E template with sector dropdown
  - [ ] Review with M&E lead; collect feedback
  - [ ] Finalize linking guide + template documentation

### For Phase 5 (Weeks 4–5, Sep 3 – Sep 17)

- [ ] **Week 4:**
  - [ ] Create `benchmark_reporting.ipynb`
  - [ ] Generate executive summary + sector profiles
  - [ ] Consolidate ZOA-unique assessment
  
- [ ] **Week 5:**
  - [ ] Build dashboard data + sample visualizations
  - [ ] Draft implementation roadmap + change management plan
  - [ ] Kickoff presentation prep + team training materials

---

## 9. Dependencies & External Inputs

| What | Owner | Timeline | Risk |
|-----|-------|----------|------|
| Proposal indicators (extracted, cleaned) | Proposal extraction pipeline | ✅ Available | None |
| M&E team feedback on linking guide | M&E lead | Week 2–3 | Delays template finalization |
| Leadership review & sign-off | Leadership | Week 5 | Delays rollout approval |
| Sector lead validation of ZOA-unique rationale | Sector leads | Week 1–4 | Strategic misalignment |
| Training schedule & resources | M&E | Week 5 | Rollout readiness |

---

## 10. Glossary

| Term | Definition |
|------|-----------|
| **Adoption Band** | Classification of ZOA indicators by how many peer organizations use them: Universal (8–9), Sector-Standard (5–7), Moderate (2–4), ZOA-Unique (0–1) |
| **Match Tier** | Matching confidence level: Tier 1 (exact slug), Tier 2 (normalized title), Tier 3 (embedding ≥0.80), No Match |
| **Alignment Score** | % of proposal indicators matched to ZOA reference: score = matched / total |
| **ZOA-Unique** | Indicator carried by ZOA but not by any peer organization (adoption = 0) |
| **Proposal-Specific** | Indicator extracted from proposal that doesn't match any ZOA indicator; may be context or project-specific |
| **Linking Guide** | Decision framework + flowchart for proposal developers to choose between ZOA indicators vs. new/proposal-specific ones |
| **M&E Template** | Excel workbook pre-populated with ZOA indicators by sector; authors customize baseline/target/MoV; includes approval workflow |
| **Benchmark** | Sector-level KPIs on ZOA library coverage, adoption distribution, and gaps vs. peers |

---

## 11. Document Version & History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-08-13 | Copilot | Initial comprehensive plan (Phases 1–5) |
| 2.0 | 2026-08-13 | Copilot | Updated: Phases 1–3 marked complete; Phases 4–5 detailed + actionable |

---

**Status:** Phases 1–3 complete ✅ | Phases 4–5 ready for execution 🚀 | Rollout planned for Q4 2026
