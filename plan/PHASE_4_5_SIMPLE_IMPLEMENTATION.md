# Phase 4 & 5: Simple Implementation Guide

**Objective:** Use LLM to link proposal indicators to ZOA reference library and generate summary tables.

**Timeline:** 1 week (straightforward implementation)

---

## Phase 4: LLM Indicator Linking

### What You Need

1. **ZOA Reference Data** (from Phase 1–3)
   - Location: `MEAL_Proposal-main/notebooks/data/zoa_indikit_net/indicators.csv`
   - Columns needed: `Indicator Name`, `Indicator Wording - ENG`, `Sector`

2. **Extracted Proposal Indicators** (existing output)
   - Location: `outputs/indicators_extracted.csv`
   - From: Existing `proposal_indicator_analysis.ipynb` extraction

### Implementation Steps

#### Step 1: Add Section to `proposal_indicator_analysis.ipynb`

Add a new markdown cell after the existing LLM extraction section:

```markdown
## Indicator Linking to ZOA Reference
Link all extracted proposal indicators to ZOA indicators using LLM.
```

#### Step 2: Load ZOA Reference

```python
import pandas as pd

# Load ZOA reference
zoa_df = pd.read_csv(
    'MEAL_Proposal-main/MEAL_Proposal-main/notebooks/data/zoa_indikit_net/indicators.csv'
)

# Keep key columns
zoa_indicators = zoa_df[[
    'Indicator Name', 
    'Indicator Wording - ENG', 
    'Sector'
]].dropna(subset=['Indicator Name'])

print(f"Loaded {len(zoa_indicators)} ZOA indicators")
zoa_indicators.head()
```

#### Step 3: Create LLM Linking Function

```python
import json

def link_indicator_to_zoa(proposal_indicator_text, zoa_reference_df, client, model_name='gpt-4o-mini'):
    """
    Match proposal indicator to ZOA reference using LLM.
    
    Returns:
        {
            'zoa_indicator': 'Indicator Name' or None,
            'status': 'Present' | 'Different' | 'Missing',
            'confidence': 0.0-1.0,
            'reason': 'explanation'
        }
    """
    
    # Build list of ZOA indicators (limit to ~30 for context window)
    zoa_list = zoa_reference_df.head(30)[['Indicator Name', 'Sector']].to_dict('records')
    zoa_text = "\n".join([
        f"  • {ind['Indicator Name']} ({ind['Sector']})" 
        for ind in zoa_list
    ])
    
    prompt = f"""
Determine if this proposal indicator matches a ZOA indicator:

PROPOSAL INDICATOR:
"{proposal_indicator_text}"

ZOA REFERENCE INDICATORS:
{zoa_text}

TASK:
Classify as ONE of:
- "Present": Exact or very similar match found in ZOA reference
- "Different": Related ZOA indicator exists but wording differs significantly  
- "Missing": No ZOA indicator covers this topic

RESPONSE FORMAT (JSON only, no extra text):
{{
    "zoa_indicator": "Exact ZOA indicator name if Found, or null if not found",
    "status": "Present",
    "confidence": 0.95,
    "reason": "Brief explanation"
}}
"""
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3  # Lower temp for consistency
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error linking '{proposal_indicator_text}': {e}")
        return {
            'zoa_indicator': None,
            'status': 'Error',
            'confidence': 0.0,
            'reason': str(e)
        }
```

#### Step 4: Apply Linking to All Indicators

```python
# Only link if LLM was used (check for existing indicators_df)
if not indicators_df.empty and OPENAI_API_KEY:
    print("Linking proposal indicators to ZOA reference...")
    
    client = LLMClient(model_name=MODEL_NAME)
    
    linking_results = []
    for idx, row in indicators_df.iterrows():
        proposal_indicator = row['indicator_text']
        
        result = link_indicator_to_zoa(proposal_indicator, zoa_indicators, client)
        
        linking_results.append({
            'index': idx,
            'indicator_text': proposal_indicator,
            'zoa_match': result.get('zoa_indicator'),
            'match_status': result.get('status'),
            'confidence': result.get('confidence', 0),
            'reason': result.get('reason', '')
        })
        
        # Print progress every 10 indicators
        if (idx + 1) % 10 == 0:
            print(f"  Linked {idx + 1} indicators...")
    
    linking_df = pd.DataFrame(linking_results)
    
    # Merge back to original dataframe
    indicators_df = indicators_df.reset_index(drop=False)
    indicators_df = indicators_df.merge(
        linking_df[['index', 'zoa_match', 'match_status', 'confidence', 'reason']], 
        on='index',
        how='left'
    )
    
    print(f"✓ Linked all {len(indicators_df)} indicators")
else:
    print("No indicators to link (LLM extraction may not have run)")
```

#### Step 5: Save Enhanced Indicator File

```python
# Save enhanced indicators with linking results
indicators_csv_linked = OUTPUTS_DIR / 'indicators_extracted_linked.csv'
indicators_df.to_csv(indicators_csv_linked, index=False)
print(f"Saved: {indicators_csv_linked}")

# Show linking stats
if not indicators_df.empty and 'match_status' in indicators_df.columns:
    print("\n=== LINKING SUMMARY ===")
    print(indicators_df['match_status'].value_counts())
    print(f"\nAverage Confidence: {indicators_df['confidence'].mean():.2f}")
```

---

## Phase 5: Generate Summary Tables

### Add New Cell: Summary by Proposal

```python
# Summary by proposal
if 'match_status' in indicators_df.columns:
    proposal_summary = indicators_df.groupby('doc_id').apply(
        lambda group: pd.Series({
            'Total Indicators': len(group),
            'Present': (group['match_status'] == 'Present').sum(),
            'Different': (group['match_status'] == 'Different').sum(),
            'Missing': (group['match_status'] == 'Missing').sum(),
        })
    ).reset_index()
    
    # Add percentages
    proposal_summary['% Present'] = (proposal_summary['Present'] / proposal_summary['Total Indicators'] * 100).round(1)
    proposal_summary['% Different'] = (proposal_summary['Different'] / proposal_summary['Total Indicators'] * 100).round(1)
    proposal_summary['% Missing'] = (proposal_summary['Missing'] / proposal_summary['Total Indicators'] * 100).round(1)
    
    # Save
    proposal_summary_csv = OUTPUTS_DIR / 'proposal_indicator_summary.csv'
    proposal_summary.to_csv(proposal_summary_csv, index=False)
    print(f"Saved: {proposal_summary_csv}\n")
    print(proposal_summary)
```

### Add New Cell: Summary by Sector

```python
# Summary by sector
if 'match_status' in indicators_df.columns and 'sector' in indicators_df.columns:
    sector_summary = indicators_df.groupby('sector').apply(
        lambda group: pd.Series({
            'Total Indicators': len(group),
            'Present': (group['match_status'] == 'Present').sum(),
            'Different': (group['match_status'] == 'Different').sum(),
            'Missing': (group['match_status'] == 'Missing').sum(),
        })
    ).reset_index()
    
    # Add percentages
    sector_summary['% Present'] = (sector_summary['Present'] / sector_summary['Total Indicators'] * 100).round(1)
    sector_summary['% Different'] = (sector_summary['Different'] / sector_summary['Total Indicators'] * 100).round(1)
    sector_summary['% Missing'] = (sector_summary['Missing'] / sector_summary['Total Indicators'] * 100).round(1)
    
    # Save
    sector_summary_csv = OUTPUTS_DIR / 'sector_indicator_summary.csv'
    sector_summary.to_csv(sector_summary_csv, index=False)
    print(f"Saved: {sector_summary_csv}\n")
    print(sector_summary)
```

### Add New Cell: Key Insights

```python
# Generate simple insights
if 'match_status' in indicators_df.columns:
    total_indicators = len(indicators_df)
    present_count = (indicators_df['match_status'] == 'Present').sum()
    different_count = (indicators_df['match_status'] == 'Different').sum()
    missing_count = (indicators_df['match_status'] == 'Missing').sum()
    
    insights = f"""
PROPOSAL INDICATOR LINKING SUMMARY
═══════════════════════════════════

Overall Statistics:
  Total proposals analyzed: {indicators_df['doc_id'].nunique()}
  Total indicators extracted: {total_indicators}
  
Linking Results:
  Present (matched to ZOA): {present_count} ({present_count/total_indicators*100:.1f}%)
  Different (similar but adapted): {different_count} ({different_count/total_indicators*100:.1f}%)
  Missing (no ZOA equivalent): {missing_count} ({missing_count/total_indicators*100:.1f}%)

Average Confidence: {indicators_df['confidence'].mean():.2f}

Top 10 Most Frequently Missing Indicators:
{indicators_df[indicators_df['match_status'] == 'Missing']['indicator_text'].value_counts().head(10).to_string()}

Top 10 Most Frequently Adapted Indicators:
{indicators_df[indicators_df['match_status'] == 'Different']['zoa_match'].value_counts().head(10).to_string()}
"""
    
    # Save insights
    insights_txt = OUTPUTS_DIR / 'insights_summary.txt'
    with open(insights_txt, 'w') as f:
        f.write(insights)
    
    print(insights)
    print(f"\nSaved: {insights_txt}")
```

---

## Output Files

After running these sections, you'll have:

| File | Contents | Use |
|------|----------|-----|
| `indicators_extracted_linked.csv` | All indicators + ZOA match status | Main output |
| `proposal_indicator_summary.csv` | By-proposal stats (Present/Different/Missing counts) | Quick overview per proposal |
| `sector_indicator_summary.csv` | By-sector stats | Which sectors need work |
| `insights_summary.txt` | Key findings in text format | Share with team |

---

## Quick Checklist

- [ ] Add "Indicator Linking" section to `proposal_indicator_analysis.ipynb`
- [ ] Load ZOA reference CSV (adjust path as needed)
- [ ] Create `link_indicator_to_zoa()` function
- [ ] Apply linking to all indicators (may take 2–5 min depending on count)
- [ ] Generate proposal summary table
- [ ] Generate sector summary table
- [ ] Generate insights text file
- [ ] Review outputs in Excel or text editor
- [ ] Done! 🎉

---

## Expected Output Example

**proposal_indicator_summary.csv:**
```
doc_id,Total Indicators,Present,Different,Missing,% Present,% Different,% Missing
Proposal_A.pdf,15,12,2,1,80.0,13.3,6.7
Proposal_B.pdf,22,16,4,2,72.7,18.2,9.1
Proposal_C.pdf,10,7,2,1,70.0,20.0,10.0
```

**insights_summary.txt snippet:**
```
PROPOSAL INDICATOR LINKING SUMMARY
═══════════════════════════════════

Overall Statistics:
  Total proposals analyzed: 3
  Total indicators extracted: 47
  
Linking Results:
  Present (matched to ZOA): 35 (74.5%)
  Different (similar but adapted): 8 (17.0%)
  Missing (no ZOA equivalent): 4 (8.5%)
```

---

**That's it! Simple, practical, no overcomplications.** 🚀
