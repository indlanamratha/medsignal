# Serious-Report Prediction Model: Decisions and Results

**Goal:** Predict whether a new FAERS report is serious, so safety teams can review likely-serious reports first. Recall matters most: missing a serious report costs more than reviewing an extra mild one.

## Data and split
- 383,296 reports with a study drug as a suspect (2020–2025), 131 features.
- Time-based split: train 2020–2023 (188,648 reports), validate 2024 (81,179), test 2025 (113,469).
- Target: `is_serious`. Serious rate falls over time: 41.0% (train), 37.5% (validation), 34.7% (test).

## Leakage controls
- Excluded all seriousness flags and the reaction outcome field.
- Excluded reaction terms that describe the outcome itself: anything containing DEATH, FATAL, HOSPITAL, DISABILITY, and COMPLETED SUICIDE.
- The 100 reaction features were chosen from training years only.

## Results (2025 test set)
| Model | PR-AUC | ROC-AUC | Precision at ~80% recall |
| --- | --- | --- | --- |
| Baseline (training rate) | 0.347 | 0.500 | 34.7% |
| Logistic regression | 0.938 | 0.959 | 92.0% |
| XGBoost | 0.953 | 0.968 | 93.7% |
| LightGBM (selected) | 0.953 | 0.969 | 94.1% |

The decision threshold was chosen on 2024 data for 80% recall; on 2025 it gave 79.5% recall.

## Country shortcut check
SHAP ranked `is_us` as the top feature. In the 2025 test set, 99.1% of non-US reports are serious, because foreign reports mostly reach FAERS only when they are serious. That is a reporting rule, not medical severity.

| Model | Evaluated on | Baseline | PR-AUC | ROC-AUC |
| --- | --- | --- | --- | --- |
| With country features | All 2025 | 0.347 | 0.953 | 0.969 |
| With country features | US-only 2025 | 0.211 | 0.848 | 0.941 |
| Without country features | All 2025 | 0.347 | 0.894 | 0.932 |
| Without country features | US-only 2025 | 0.211 | 0.829 | 0.935 |

**Conclusion:** The model learns real severity signals: on US-only reports it reaches 0.848 PR-AUC against a 0.211 baseline. That is the headline metric. Country stays in the deployed model because it is known when a report arrives, but performance is reported separately for US reports.

## Other top features (SHAP)
Number of reactions, consumer reporter (milder reports), medication-error and injection-site terms (usually not serious), and lactic acidosis, stroke, and pancreatitis (usually serious).

## Limitations
- Seriousness is assigned by the reporter and can be inconsistent.
- Batch submissions (for example semaglutide, July 2025) shift the test set toward mild reports.
- Rule-based term filters stand in for the licensed MedDRA hierarchy.