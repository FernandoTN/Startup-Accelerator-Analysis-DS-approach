# Startup Accelerator Analysis — Data Science Approach

Short description
This repository contains a data-science-driven analysis of GALI (Global Accelerator/Incubator) program data. The primary deliverable is the Jupyter notebook `GALI_Integrated_Final_Analysis.ipynb` which performs the data cleaning, exploratory analysis, feature engineering and initial modeling used to understand accelerator outcomes.

Contents (what's in this repo)
- GALI_Integrated_Final_Analysis.ipynb — Main analysis notebook (primary artifact).
- GALI_External_DataRelease_2020.xlsx — Raw source Excel dataset (~27 MB).
- convert_gali_excel.py — Helper script to parse/convert the Excel into analysis-ready CSV(s).
- RP_Info.pdf — Reference / supporting document (project / research plan info).
- RP_Info_extracted.txt — Text extracted from RP_Info.pdf (convenience for search).
- TodoFixes.md — Notes and TODO items for future improvements.
- Claude.md — Notes or prompts used during research.
- data/ — Suggested location for processed data (empty in repo).
- submission/ — Placeholder for outputs destined for submission (empty).
- tmp_files/ — Temporary working files (empty).
- Legacy/ — Older files or archived work.

Why this repo
The goal is to provide a reproducible, documented data-science workflow for analyzing accelerator cohorts and their outcomes using the provided GALI dataset. The notebook walks from raw data ingestion through cleaning, descriptive analyses and initial predictive experiments.

Quick start (recommended)
1. Clone the repository
   ```bash
   git clone https://github.com/FernandoTN/Startup-Accelerator-Analysis-DS-approach.git
   cd Startup-Accelerator-Analysis-DS-approach
   ```

2. Set up environment (example using pip)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install jupyterlab pandas numpy matplotlib seaborn scikit-learn openpyxl xlrd
   ```
   - If you prefer conda:
   ```bash
   conda create -n gali-analysis python=3.10
   conda activate gali-analysis
   pip install jupyterlab pandas numpy matplotlib seaborn scikit-learn openpyxl xlrd
   ```

3. Prepare data
   - The raw Excel file is `GALI_External_DataRelease_2020.xlsx`. It is relatively large (~27 MB).
   - To create analysis-ready CSV(s) run the provided script:
     ```bash
     python convert_gali_excel.py --input GALI_External_DataRelease_2020.xlsx --output data/gali_processed.csv
     ```
     (See `convert_gali_excel.py` header/usage for exact CLI options.)

4. Open and run the notebook
   ```bash
   jupyter lab
   # then open GALI_Integrated_Final_Analysis.ipynb
   ```
   - Run cells sequentially. The notebook contains narrative, figures and code cells that reproduce the analyses.

Files explained (more detail)
- GALI_Integrated_Final_Analysis.ipynb
  - Overview of dataset schema.
  - Data cleaning and sanity checks.
  - Exploratory Data Analysis (EDA) — summary tables and visualizations.
  - Feature engineering for modeling (if applicable).
  - Baseline predictive experiments and evaluation metrics.
  - Conclusions and future work.

- convert_gali_excel.py
  - Script to convert multi-sheet Excel into flat CSV(s) and optionally to clean/normalize columns.
  - Use it to avoid re-running slow Excel parsing inside the notebook.

- RP_Info.pdf / RP_Info_extracted.txt
  - Project plan, metadata or documentation referenced by the analysis.
  - `RP_Info_extracted.txt` is a convenience plain-text extraction for quick searching.

- TodoFixes.md
  - Short list of fixes, improvement ideas and outstanding tasks.

Reproducibility notes
- Notebook outputs (plots, intermediate CSVs) are not committed to keep repo small — run the notebook locally to reproduce visuals.
- If you add or change heavy intermediate files, place them under `data/` or `tmp_files/` and add to `.gitignore` if needed.

Dependencies (high-level)
- Python 3.8+
- jupyterlab / jupyter notebook
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- openpyxl (Excel reading/writing)
- xlrd (if older Excel formats used)

Contact / Author
- Repository owner: FernandoTN
- For questions about the analysis or data, open an issue in this repository.
