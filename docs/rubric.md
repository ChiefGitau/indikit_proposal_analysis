# Match rubric (version rubric-v1)

How a document indicator relates to a ZOA IndiKit indicator. The text between the markers is loaded into the
LLM prompt by `pipeline/agent/rubric.py`; the version is `config.RUBRIC_VERSION` and is part of every cache key.

<!-- prompt-start -->
Match types:
- verbatim: the document indicator measures the same thing as the IndiKit indicator with (nearly) identical wording. Differences are limited to the target value, placeholders filled in (the target group, the area, the year) or word order.
- close: the document indicator measures the same thing with different wording, a different unit (number instead of percentage), a different population or a narrower / wider scope.
- absent (none): nothing in the evidence measures the same thing. The same topic or sector is not enough: an indicator about water points does not match an indicator about handwashing.
- not an indicator: activities, deliverables, assumptions, budget lines, or targets without a measure (for example "Train 30 farmers", "5 VSLAs supported", "Latrines constructed") are never matches, even when they share words with the IndiKit indicator.

Worked examples:
| Document text | IndiKit indicator | Label | Why |
|---|---|---|---|
| "Number of people having access to basic, safe and dignified shelter solutions" | 5179 Access to Adequate Shelter | verbatim | Same measure, same wording |
| "Number of local people involved in voluntary or paid work on shelter (re)construction" | 5181 Local Participation in Shelter Construction | verbatim | Same measure, same wording |
| "% of households with soap and water at a handwashing station" | 5138 Handwashing Facilities With Soap and Water | verbatim | Same measure |
| "75% of trained PIP farmers report increased yields" | 5260 Agricultural Productivity | close | Same outcome, narrower population, self-reported |
| "Number of functional water points constructed or rehabilitated" | 5129 Water Source Functionality | close | Counts water points; IndiKit measures the share that is functional |
| "Number of people receiving multi-purpose cash" | 5239 Number of Cash Transfer Recipients | close | Same count, restricted to one modality |
| "5 VSLAs supported" | (none) | none | Activity count; self-help-group indicators measure functionality or coverage, not groups supported |
| "70% reduction in seasonal migration" | (none) | none | No IndiKit indicator measures migration |
| "Train 30 farmers on climate-smart agriculture" | (none) | not an indicator | An activity |

One-to-many rule: a document indicator may match more than one IndiKit indicator, and one IndiKit indicator may match several document indicators. Report every match; the pipeline marks one as primary (verbatim before close, then the higher retrieval score) and keeps the others as alternates.
<!-- prompt-end -->

## Notes for reviewers

- "Same measurement" means the numerator and the population are the same kind of thing, not that the sector is the same.
- A target in the document ("60% by 2026") does not change the match type; it is recorded as the target value.
- When a document row holds several indicators, each is judged separately.
- The example IDs were checked against `pipeline/metadata/indikit_metadata.jsonl` on 2026-11-12: all nine names match the
  metadata. 5239 is an export-only indicator (`on_site = false`); it is kept because the example is about the modality rule.
