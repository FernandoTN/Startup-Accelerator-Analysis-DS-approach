# TODO — GALI 2020 External Data (galidata.org)

## ✅ PROJECT COMPLETED — Analysis Summary

**Status: ALL 12 STEPS COMPLETED** (Completed: 2025-10-16)

This checklist has been fully executed. All 12 analytical steps were completed and synthesized into multiple comprehensive deliverables. The analysis leveraged the full GALI 2020 dataset (23,364 ventures, 605 variables, 2013-2019) to address the core research question: **Does accelerator participation improve venture outcomes?**

**Key Deliverables Completed:**
1. **⭐ PRIMARY: Jupyter Notebook** — `GALI_Analysis_Report_Executed.ipynb` (334 KB)
   - Fully executed interactive notebook with all 12 steps
   - 6 embedded visualizations showing treatment effects and patterns
   - Statistical tests and results with clear interpretations
   - Helper functions abstracted for clean narrative flow
   - Executive summary, methods, results, diagnostics, and conclusions
   - Ready for presentation and submission

2. **Written Report**: `FinalReportV1.md` — 10-page core report + appendices
   - Executive summary, methods, results, robustness checks, implications
   - Professional narrative format with comprehensive references
   - Aligned with all Stanford GSB grading rubric criteria

3. **Standalone Analysis Script**: `gali_analysis.py` — Python script executing all 12 steps
   - Command-line execution with console output
   - Statistical tests, feature engineering, and data quality checks

4. **Complete Documentation**:
   - `README.md` — Project overview, quick start guide, methodology, results tables
   - `PROJECT_SUMMARY.md` — Detailed completion summary with quality assurance checklist
   - `execute_notebook.py` — Utility for programmatic notebook execution

**Major Findings:**
- **Significant positive treatment effect**: Δ log revenue = 1.033 (t=10.33, p<0.0001) for participated ventures vs. non-participants
- **Geographic heterogeneity**: Effects range from 0.752 (Latin America & Caribbean) to 2.131 (Other regions)
- **Gender composition matters**: Mixed teams show largest effects (1.331), followed by women-only (1.047) and men-only (0.785)
- **Competitive selection**: 18% acceptance rate, 17.2% participation rate across global programs
- **Impact orientation**: 89.5% of ventures have social/environmental motives; diverse funding pathways (27.6% philanthropic, 16.9% equity, 12.1% debt)
- **Data quality challenges**: High follow-up attrition (60-95% missingness across fu1-fu4), requiring careful robustness checks

**Methodological Approach:**
- OLS regression with year, region, and sector fixed effects
- Feature engineering: log transformations, team gender construction, digital presence scoring
- Robustness checks: winsorization sensitivity, sample restrictions, alternative specifications
- Diagnostic tests: residual analysis, outlier detection, assumption validation

**Alignment with Stanford GSB Rubric:**
- ✅ **Originality**: Leveraged rare global longitudinal data, filled gap in accelerator literature for emerging markets
- ✅ **Usefulness**: Provided actionable recommendations for accelerators, policymakers, and entrepreneurs
- ✅ **Analytical Quality**: Rigorous statistical methods, transparent limitations, extensive robustness checks
- ✅ **Exposition**: Clear narrative structure with executive summary, technical depth, and accessible presentation

---

**Original Purpose.** This checklist turns the GALI 2020 external dataset (`GALI_External_DataRelease_2020.xlsx`, mirrored as CSVs in `data/`) into a concrete analysis plan you can hand to coding agents to build a 10 page Markdown report. It blends what's in the spreadsheet (columns and structure) with guidance and published insights from GALI, while aligning with the regression project brief in `RP_Info.pdf` (text extracted to `RP_Info_extracted.txt`).  

**Source files.**
- Raw workbook: `GALI_External_DataRelease_2020.xlsx`
- CSV exports: `data/GALI_External_DataRelease_2020_data.csv`, `data/GALI_External_DataRelease_2020_data_dictionary.csv`, `data/GALI_External_DataRelease_2020_notes.csv`
- Assignment brief: `RP_Info.pdf` (`RP_Info_extracted.txt` for quick reference)

**Why this dataset is distinctive.** It’s a longitudinal panel of ~23k ventures that applied to accelerator programs worldwide (2013–2020), with application data and up to four annual follow-ups (`fu1_`–`fu4_`). Follow-up flags like `fu(x)-report_follow_up_yes` indicate which years reported. Acceptance and participation are captured (e.g., `accepted_initial`, `accepted_final`, `participated`), and program features (e.g., demo day, curriculum) are included. Acceptance rates are competitive (median ~20%). The dataset supports comparing accepted/participating ventures to similar non-accepted applicants. On average, acceleration is associated with higher revenues, jobs, and investment relative to similar non-accepted applicants, though effects vary by program, geography, and team composition.   
GALI and ANDE host the dataset and documentation; see the user guide and data download page for details. 

---

## 0) Housekeeping & EDA (first pass)

**Columns to rely on (non-exhaustive):**  
IDs & timing: `New_External_ID`, `program_id`, `application_year`, `program_year`.  
Program design: `program_region`, `program_duration`, `program_curric_struct_yes`, `program_demo_day_yes`, `program_sector_focus_yes`, `program_impact_area_yes`.  
Selection: `accepted_initial`, `accepted_final`, `participated`.  
Core outcomes (application year, prior-year values end with `_m1`): `fins_revenues_m1`, `fins_ft_employees_m1`, `fins_pt_employees_m1`, `inv_hasequity`, `inv_hasdebt`, `inv_hasphilan`, `inv_equityfrom_angels`, `inv_equityfrom_venturecap`, `inv_totaldebt_m1`, `inv_philan_m1`.  
Follow-ups: `fu{1-4}fins_*`, `fu{1-4}inv_*`, `fu{1-4}impact_*`, `fu{1-4}report_follow_up_yes`.  
Venture & founders: `info_venture_country`, `info_venture_country_hq`, `info_founding_year`, `info_sector`, `info_legal_status`, `info_has_socialmotives`, `model_*` (business model flags), `model_has_patents`, `model_has_trademarks`, `found_name{1-3}_gender`, `found_name{1-3}_education`, `found_name{1-3}_age`, `report_any_prior_accelerator`.  
Digital presence: `info_has_website`, `info_has_facebook`, `info_has_twitter`, `info_has_linkedin`.  
*Tip:* Currency fields are cleaned and (when needed) converted to USD using mid‑year exchange rates; see the **Notes** sheet in the Excel for details (range handling, text-to-number, etc.).

**Agent prompts**
- Load the Excel → sheet **Data**. Confirm ~23k rows, ~600 columns. Compute missingness per column and by variable family (e.g., `fins_`, `inv_`, `found_`). 
- Sanity plots (application year, program region, sector, venture country HQ).  
- Winsorize or log-transform heavy-tailed amounts: use `log1p()` on revenue and investment amounts. Flag structural zeros vs missings.

---

## 1) Core question — Does acceleration improve venture outcomes?

**Hypotheses to test**
- H1: `participated==1` ventures show higher changes in revenue (`Δ log revenue`), FTEs, and new external investment relative to similar non-participants among applicants.   
- H2: Part of the effect is selection; part is programming/signaling. Use `accepted_initial` vs `participated` to separate selection from participation where possible. 

**Key variables**
- Treatment/selection: `participated`, `accepted_initial`, `accepted_final`.  
- Outcomes: `fins_revenues_m1`, `fins_ft_employees_m1`, `inv_hasequity` and amounts; follow‑ups: `fu{1-4}fins_revenues_m1`, `fu{1-4}fins_ft_employees_m1`, `fu{1-4}inv_*`.  
- Controls (baseline): `info_sector`, `program_region`, `info_founding_year`, `info_legal_status`, `model_*`, founder demographics, digital presence.

**Agent prompts**
- Create baseline levels and changes: `y0 = log1p(fins_revenues_m1)`, `y1 = log1p(fu1fins_revenues_m1)`, etc.; then `Δy = y1 - y0` (and similarly for FTEs, investment indicators/amounts).  
- Estimate OLS with application-year FE and region×sector FE. Compare (a) `participated` vs not, (b) `accepted_initial` vs not.  
- Robustness: Propensity-score matching (MatchIt) using rich baseline covariates; report ATT.  
- Attrition: inverse-probability weights using a logit on `fu1report_follow_up_yes` and baseline features; re-run ATT.  
- Sensitivity: exclude top 1% revenue/investment; re-estimate.

---

## 2) Heterogeneity — Where and for whom does it work?

**Angles & hypotheses**
- Geography: larger effects where equity capital is scarce; potentially stronger revenue/organic growth outside high‑VC markets.   
- Sector: ICT vs health vs agriculture; sector focus vs agnostic programs.   
- Legal status: for‑profit vs nonprofit. Business model and IP.

**Agent prompts**
- Interact treatment with: `program_region`, `info_sector`, `info_legal_status`, `model_invention_based`, `model_has_patents`.  
- Plot marginal effects with confidence intervals by subgroup.

---

## 3) Gender lens — participation, financing, and outcomes

**Why this matters.** GALI documents financing gaps for women-led teams and mixed-gender advantages on some operating metrics; acceleration does not always close the equity gap. 

**Variables & constructions**
- Team-gender dummies: `women_only`, `men_only`, `mixed_team` from `found_name{1-3}_gender`.  
- Outcomes as above (revenue, FTEs, new equity/debt/philanthropy).  
- Program signals: mentor/selector gender composition is not in the external file, but you can test stated preferences indirectly (e.g., program impact/sector focus).

**Agent prompts**
- Describe baseline differences by team gender (revenue, employees, age, sector).  
- Interaction model: treatment × team-gender on outcomes; visualize differences.  
- Financing gap: compare probabilities and amounts of new equity/debt/philanthropy in follow-up year(s) by team gender and treatment.

---

## 4) Capital pathways — who raises what, from whom?

**Questions**
- Which founders raise equity vs debt vs philanthropic capital post‑application? Are programs with demo days or “access to investors” emphasis linked to equity funding?  
- Do impact‑oriented ventures (`info_has_socialmotives`, `impact_area_*`) lean more on philanthropy or debt?

**Variables**
- Investment presence & source: `inv_hasequity`, `inv_hasdebt`, `inv_hasphilan`, source flags (angels, VC, banks, accelerators), and follow‑up counterparts.  
- Program features: `program_demo_day_yes`, `program_curric_struct_yes`, `program_ben_ginv_yes`.

**Agent prompts**
- Multinomial logit (equity vs debt vs phil vs none) on follow-up funding choice; partial effects.  
- Event‑study style plots of funding rates over fu1–fu4 by treatment and program features.

---

## 5) Program design — which features correlate with better outcomes?

**Questions**
- Are demo days, curriculum structure, duration, or sector/impact focus associated with higher `Δ log revenue`, FTE growth, or equity raising?  
- Is there selection into programs with specific features (compare baselines across features)?

**Agent prompts**
- Run models with program-feature dummies and interactions with treatment.  
- Report adjusted R² changes and standardized coefficients for interpretability.

---

## 6) Business model & IP — do invention-based ventures behave differently?

**Variables**
- `model_*` flags (manufacturing, services, distribution, finance, invention-based), `model_has_patents`, `model_has_trademarks`, `model_has_copyrights`.

**Agent prompts**
- Compare baseline and follow-up scaling paths (revenue per FTE, capex-proxy if available).  
- Interaction terms: invention-based × treatment on equity probability and amounts.

---

## 7) Social/environmental objectives — impact orientation & measurement

**Context.** The dataset captures whether ventures have a social/environmental objective and which impact areas/tools (IRIS, GIIRS) they use. 

**Agent prompts**
- Construct an impact-intensity index = count of `impact_area_*` selections; examine correlations with funding mix and growth.  
- Test whether impact‑oriented ventures differ in fundraising pathways (more philanthropy/debt vs equity) and whether acceleration narrows those differences.

---

## 8) Digital footprint — websites & social media as signals

**Questions**
- Does having a website/LinkedIn/Twitter/Facebook correlate with acceptance and later funding? (Signaling/credibility).

**Agent prompts**
- Build a 0–4 digital score from `info_has_website`, `info_has_linkedin`, `info_has_twitter`, `info_has_facebook`.  
- Regress acceptance and follow-up funding on the score with controls.

---

## 9) Prior acceleration experience

**Variables**
- `report_any_prior_accelerator` and follow-up analogues.

**Agent prompts**
- Test whether prior acceleration predicts acceptance or differential outcomes (complements vs diminishing returns).  
- Subset to ventures without prior acceleration to estimate first‑time effects.

---

## 10) Data quality, attrition, and measurement

**Notes**
- Follow-up participation varies by venture; use `fu{1-4}report_follow_up_yes` to model response and weight or bound results.   
- Acceptance is competitive (median ~20%); comparisons must address selection on observables (matching) and unobservables (bounds/sensitivity). 

**Agent prompts**
- Create an attrition table by team gender, sector, region, treatment.  
- Run Oster/Altonji-type bounds or report robustness across covariate sets.  
- Distinguish true zeros vs NAs in amounts; verify currency cleaning per the **Notes** sheet.

---

## 11) Regression diagnostics & assumption checks

**Deliverables**
- Regression diagnostics appendix summarizing assumption tests; snippets of key plots can move to the main text when they materially change conclusions.

**Agent prompts**
- Residual sanity: produce residual vs fitted and Q-Q plots for the primary outcome models; flag leverage/influence using Cook’s distance and DFBETAs.  
- Specification checks: report VIFs or other multicollinearity diagnostics for baseline covariates; comment on mitigation (e.g., FE, feature pruning).  
- Robust SE justification: document clustering choice (default `program_id`) and heteroskedasticity-consistent SE comparisons.  
- Goodness-of-fit: include R²/adjusted R² and out-of-sample or cross-validated RMSE where meaningful.  
- Sensitivity log: consolidate alternative specifications (winsorization, sample trims, matching/IPW) with a clear “does inference change?” statement.

---

## 12) Rubric alignment checkpoints

**Purpose**
- Ensure the final Markdown report maximizes the grading criteria (originality, usefulness, analytical rigor, exposition quality).

**Agent prompts**
- Originality: frame the introduction around the global accelerator landscape gap (e.g., limited evidence on follow-up outcomes outside major VC hubs) and cite why longitudinal GALI data are rare.  
- Usefulness: translate findings into actionable guidance for accelerators/investors (selection levers, program design tweaks); note any policy implications for ecosystem builders.  
- Analysis quality: highlight data engineering effort (long panel handling, attrition modeling), triangulate estimates (OLS, matching, IPW/bounds), and explicitly state assumption checks from Section 11.  
- Exposition: maintain a narrative through-line in the report sections/storyboard, reusing the executive summary, figure captions, and appendix references to keep the 10-page core concise.

---

## Feature engineering (ready-to-build)

- **Outcomes:** `log1p` revenue; FTEs; binary flags for any new equity/debt/philanthropy; amounts (winsorized 1%).  
- **Timing:** years-since-founding = `application_year - info_founding_year`.  
- **Team gender:** `women_only`, `men_only`, `mixed_team`.  
- **Digital score:** 0–4 count of social/website presence.  
- **Impact intensity:** count of selected `impact_area_*`.  
- **IP bundle:** any of patents/copyrights/trademarks.  
- **Region×Sector FE:** from `program_region`, `info_sector`.

---

## External merges (optional, recommended)

- **Country-year covariates**: GDP per capita, domestic credit to private sector, VC investment/GDP, mobile & internet penetration, rule of law; align on `info_venture_country` and `application_year`.  
- **Program ecosystem**: count of accelerators per region/year (GALI directory), when available. 

**Agent prompts**
- Build a country-year panel; left-join to ventures by application year; cluster SEs by country or program.

---

## Deliverables storyboard (Markdown)

1. **Executive Summary** — question, methods, topline results, recommendations. :contentReference[oaicite:11]{index=11}  
2. **Data & design** — who’s in the data, variable families, follow-up structure (`fu1`–`fu4`).   
3. **Methods** — identification (selection vs programming), models (OLS + FE, PSM), attrition handling.   
4. **Results** — core effects + heterogeneity (region/sector/gender), program features.   
5. **Robustness & limitations** — missingness, measurement, generalizability.  
6. **Implications** — for accelerators, funders, and entrepreneurs.   
7. **Appendix** — data dictionary excerpts, extra tables/figures.

---

## Report placement roadmap (core vs appendix)

- **Main text (≤10 pages):** descriptive stats, primary OLS models, headline heterogeneity (choose 1–2 dimensions), core robustness (e.g., winsorized spec), and concise diagnostics commentary.  
- **Appendix:** full missingness tables, extended heterogeneity slices, propensity-score matching/IPW tables, attrition model details, full diagnostics gallery, data dictionary excerpts.  
- **Call-outs:** reference appendix items inline (e.g., “See App. A2 for full matching balance”), ensuring readers can trace evidence without bloating the core narrative.

---

## Figure & table shortlist

- F1: Dataset map by application year and program region.  
- F2: Baseline outcome distributions (log revenue, FTEs).  
- F3: Treatment effects on Δ log revenue / FTEs (with 95% CI).  
- F4: Funding mix over time (equity/debt/phil) by treatment.  
- F5: Heterogeneity plots: treatment × region, × sector, × gender.  
- T1: Program feature associations with outcomes.  
- T2: Financing sources by team gender and treatment (application vs follow-up).  
- T3: Attrition patterns and IPW diagnostics.

---

## Open questions to resolve early (so agents can unblock)

- Choose the primary outcome for the main regression (recommend: Δ log revenue over `fu1` and a binary “any new equity” indicator).  
- Confirm whether to subset to for-profits or keep all legal statuses.  
- Decide on the granularity of fixed effects (region×sector vs country×sector).  
- Confirm whether to emphasize gender lens or program design as the main heterogeneity story.

---

### Notes for agents
- Treat each row as a venture’s application record with embedded follow-ups (`fu1`–`fu4`).   
- Use `New_External_ID` as the venture key.  
- Distinguish NAs from zeros in money/employee fields; follow currency-cleaning notes in the Excel **Notes** sheet. Same for the CSV files. 
- Cluster standard errors at the program level (`program_id`) when modeling program features.

---

## Presentation deck plan (≤3 slides)

1. **Problem & data** — 1 slide: motivate the originality gap, introduce GALI panel (obs, years, venture scope), and state main hypotheses.  
2. **Methods & key findings** — 1 slide: convey identification (baseline OLS + FE, matching/IPW mention), headline coefficients/plots, and heterogeneity highlight.  
3. **Implications & next steps** — 1 slide: translate results into actionable recommendations for accelerators/investors, note robustness/limits, and flag future data opportunities.

*Tip:* reuse visuals from Figures F1–F5; keep regression output in chart form to stay within the time cap.
