# FIXES Roadmap: GALI Analysis Notebook

## 1. Data Integrity & Feature Engineering
- **Stop zero-imputation of follow-up revenue (`fu1fins_revenues_m1`)**; preserve `NaN` to reflect attrition and recompute `delta_log_revenue` only on valid pairs.
- Audit all engineered variables to ensure imputations align with variable meaning (e.g., separate binary flags vs continuous values).
- Document variable families retained vs discarded, with rationale and missingness thresholds.

## 2. Attrition & Missing Data Strategy
- Build explicit follow-up response indicators before feature engineering.
- Produce attrition tables/visuals by treatment status and key covariates; model response probability (logit/probit).
- Implement sensitivity analyses (complete-case vs imputed vs weighted) and report impact on outcomes.

## 3. Outcomes & Diagnostics
- Recompute outcome distributions after fixing imputations; add histograms/density plots with interpretation.
- Provide summary stats (mean, median, SD, 95% CI) for revenue, employment, investment at baseline and follow-up.
- Flag outliers and consider winsorization/trimming with transparent thresholds.

## 4. Causal Identification Improvements
- Estimate treatment effects using regression with controls (year, region, baseline size, sector) and report coefficients with CIs.
- Add propensity score matching/weighting or doubly-robust estimators to address selection bias.
- Report covariate balance diagnostics before/after adjustments.

## 5. Heterogeneity Analyses
- Formalize sub-hypotheses (region, gender, funding, program design) with expected directions.
- For each subgroup, show sample sizes, standard errors, and multiple-testing adjustments (e.g., Bonferroni or BH).
- Visualize heterogeneous effects via coefficient plots or faceted charts with clear legends.

## 6. Additional Visuals & Correlations
- Covariate correlation heatmap; highlight key relationships.
- Baseline vs follow-up scatterplots by treatment status.
- Temporal trends (application year) of participation, outcomes, and baseline metrics.
- Attrition-by-treatment plots and balance plots of standardized mean differences.

## 7. Communication & Documentation
- Reconcile all reported statistics (e.g., treatment effect values) to avoid inconsistencies.
- Update executive summary and conclusions to reflect methodological caveats and revised estimates.
- Add figure/table captions with concise interpretations; remove unsupported statements (e.g., un-run winsorization claims).
- Maintain academic tone while keeping readability (minimal emojis, precise language).

## 8. Deliverables Checklist
- [ ] Revised feature engineering script with correct handling of missing follow-up data.
- [ ] Attrition analysis notebook section with visuals and commentary.
- [ ] Regression/matching results with diagnostics and effect tables.
- [ ] Updated heterogeneity section with formal sub-hypotheses and adjusted inference.
- [ ] Enhanced visualization suite (balance plots, scatterplots, timelines).
- [ ] Polished executive summary and conclusions consistent with evidence.
- [ ] Updated documentation of dataset variables and selection decisions.
