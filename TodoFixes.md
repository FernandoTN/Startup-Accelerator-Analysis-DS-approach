# Roadmap to Strengthen GALI_Integrated_Final_AnalysisV7.ipynb

## 1 Immediate Reporting Corrections
- [x] Update the H1b figure takeaway so treated/control counts match the table outputs, especially for ICT and Environment cohorts (`tmp_files/GALI_analysis.md:4568-4577`, `tmp_files/GALI_analysis.md:4613`).  
- [x] Reconcile all references to the employment p-value so the executive summary, scorecard, and conclusion use the same number and rationale (`tmp_files/GALI_analysis.md:8718-8722`, `tmp_files/GALI_analysis.md:8754-8769`).
- [x] Replace shorthand p-value notes such as “p = 0.00” with precise rounded statistics in the hypothesis scorecard for transparency (`tmp_files/GALI_analysis.md:8705-8723`).
- [x] Fix the missing information for section 1.4
- [x] Fix the order for sections 10.1, there is a duplicate, fix the numbering an make sure the consclusions for thos sub sections match ocrrectly
- [x] Section 10.2 & 10.2a show in the results the results of section 10.3 Then also section 10.3 & 10.4 are erroneously shown after the start of section 11 and needs to be brought back to their correct position.
- [x] Cook's distance section after section 11.2 is not running correctly and the conclusions mention 11.3 but the original title is nowhere to be found, fix it.
- [x] Section 11.6 is missing a results & interpretation section at the end.
- [x] Add a brief results & interpretation section after Hypothesis score card at the end.
- [x] fix the numbering of the technical appendix.

## 2 Analytical Clarifications
- [x] Expand the H3 discussion to explain why the raw diff-in-diff shows a −1.7 pp convergence while the regression implies a +2.1 pp lift; replicate both calculations and identify the drivers (controls, weighting, or sample differences) before updating the text (`tmp_files/GALI_analysis.md:5813-5815`).  
- [x] Formalise the justification for classifying H2 as “directional” despite a highly significant OLS/IPW estimate: either provide robustness evidence that overturns it or adjust the verdict to match the statistics (`tmp_files/GALI_analysis.md:8716-8722`, `tmp_files/GALI_analysis.md:8754-8769`).
- [x] Document how the strict-caliper PSM handles unmatched treated ventures (2% left out) and clarify whether employment effects truly “evaporate” in that design by showing the full ATT table and confidence interval (`tmp_files/GALI_analysis.md:3548-3576`, `tmp_files/GALI_analysis.md:8769`).

## 3 Presentation & Narrative Improvements
- [x] Add a short methods footnote in Section 2 describing the bootstrap routine used for PSM standard errors so readers can assess precision assumptions (`tmp_files/GALI_analysis.md:3406-3482`).  
- [x] Provide a compact summary table that links each hypothesis to the specific model, sample size, effect, and significance level to supplement the prose-heavy narrative (executive summary through Section 12).  
- [x] Streamline repeated “What we’re exploring / Methods / Relevance” paragraphs by collapsing them into a standard intro box per section to reduce repetition of inline titles without losing rubric alignment.
