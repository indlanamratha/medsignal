# MedSignal: Key Findings (Week 2 Analytics)

**Data:** FDA Adverse Event Reporting System (FAERS) via the openFDA API, reports received January 2020 to December 2025.
648,877 raw records were deduplicated to 543,676 unique reports. 383,296 of them list one of the 13 study drug groups as a **suspect** drug; that is the analysis set used below.

**Important:** A FAERS report shows that a reaction was reported after a drug was taken. It does not prove that the drug caused it. These findings are signals worth investigating, not conclusions about risk.

---

## 1. Who reports a drug changes how "serious" it looks

**Finding:** Drugs reported mostly by consumers have far lower serious-report rates. Tirzepatide reports are 94.3% from consumers and 15.7% serious; metformin reports are 24.0% from consumers and 85.1% serious.

**Evidence:** `queries/05_reporter_mix_by_drug.sql`; dashboard "Reporting patterns" tab.

**So what:** Raw report counts and serious rates cannot be compared across drugs as measures of risk. Safety comparisons need disproportionality statistics (PRR/ROR) that account for each drug's overall reporting pattern.

**Caveat:** Reporter type is self-reported and sometimes missing.

## 2. The biggest report spikes were batches of mild reports, not new safety problems

**Finding:** Semaglutide reports jumped in July 2025, driven by 7,916 consumer reports, of which only 19.6% were serious. In the months before and after, consumer reports were about 95% serious. Naltrexone/bupropion showed the same pattern in November 2025: 494 consumer reports, only 3.6% serious.

**Evidence:** `queries/09_spike_root_cause.sql`; dashboard "Spike investigator".

**So what:** A monitoring rule based only on monthly report counts would raise false alarms. Spike alerts should also check the serious-report rate and reporter mix before escalating.

**Caveat:** The data does not show who submitted the batch. The pattern is consistent with a bulk submission, such as a backlog sent at once.

## 3. The weight-loss drug boom is visible in reporting volume

**Finding:** Tirzepatide reports grew from 4 in 2020 to 59,369 in 2025, and semaglutide from 2,840 to 16,469. Dulaglutide, an older GLP-1, fell from 9,364 to 4,740. Naltrexone/bupropion (about 6x) and phentermine (about 4x) also grew.

**Evidence:** `queries/01_reports_by_drug_and_year.sql`; dashboard "Overview" tab.

**So what:** Report volume tracks how many people use a drug, so newer, fast-growing drugs dominate the database. Safety teams should expect rising volume and plan review capacity for the newest GLP-1 / GIP therapies.

**Caveat:** Growth in reports reflects growth in use and awareness, not necessarily a change in safety.

## 4. GLP-1 drugs show a distinct stomach, bowel, and injection-site pattern

**Finding:** Compared with other diabetes drugs (metformin, SGLT2, DPP-4, insulin), several reactions appear far more often in GLP-1 / GIP reports: gastrointestinal hypomotility (33x), impaired gastric emptying (19x: 1.71% vs. 0.09% of reports), ileus (15x), and injection-site reactions (15-35x). Optic ischaemic neuropathy, a rare eye condition, appears 21x more often (0.21% vs. 0.01%).

**Evidence:** `queries/06_glp1_vs_other_diabetes_reactions.sql`.

**So what:** These are the candidates to test formally with PRR/ROR in Week 3 and to check against official drug labels in Week 5.

**Caveat:** This is a simple ratio of reporting percentages, not a statistical test. Very small comparator percentages can inflate the ratio.

## 5. Data quality issues that any analysis must handle

**Finding:**
- Medication-error terms rank among the top "reactions" for dulaglutide: incorrect dose administered (8.7% of reports), inappropriate schedule of administration (6.4%), and extra dose administered (5.6%). These describe how the product was used, not a medical reaction.
- "Death" is recorded as a reaction in 22.7% of dapagliflozin reports. It is often an outcome, not a specific reaction, and needs separate handling.
- Metformin and sitagliptin are the most common pair of suspect drugs (4,359 reports), partly because combination products match both drugs.

**Evidence:** `queries/02_top_reactions_per_drug.sql`, `queries/07_drugs_reported_together.sql`; dashboard "Drug explorer" toggle.

**So what:** Signal detection in Week 3 must separate medication errors from medical reactions, treat death as an outcome, and account for combination products. Otherwise results will be distorted.

---

## Limitations

- FAERS is voluntary for patients and healthcare professionals, so many events are never reported.
- Reports can be duplicated across submitters; deduplication here is by report ID and version.
- The analysis covers 13 drug groups only, so "other drugs" comparisons use this set, not all of FAERS.