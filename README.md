# VaxGap: Predicting Missed-Vaccination Risk in Children

A machine learning pipeline and interactive dashboard that predicts which children aged 12 to 23 months are at risk of **not completing their basic vaccinations**, using Demographic and Health Survey (DHS) data, and explains which factors drive that risk.

The project was inspired by my experience as a polio vaccination volunteer in rural Pakistan, where records were kept on paper and nobody could quickly tell which areas had been missed. The question behind it: *can survey data tell us where a vaccination team should go before the day starts, rather than after it ends?*

> **Important: current status of the data**
> All results in this repository were produced on the **DHS Program model (practice) dataset**, which is synthetic and does **not** represent any real country. The numbers below show that the pipeline works. They are **not findings about Pakistan**. The same code is being applied to the Pakistan DHS 2017-18 Children's Recode, and this README will be updated when those results are available.

---

## What it does

1. **Builds the target.** A child counts as *basic vaccinated* if the survey records all of: BCG, DPT 1-3, Polio 1-3 and Measles. The model predicts the opposite event, `missed = 1` (at least one vaccine missing).
2. **Trains and compares models** with 5-fold stratified cross-validation: a Dummy baseline, Logistic Regression and Random Forest.
3. **Explains the model** with SHAP values and logistic-regression coefficients.
4. **Checks fairness across groups** by reporting performance separately for urban/rural children and for each wealth quintile.
5. **Serves a dashboard** (Streamlit) with coverage charts, driver analysis, a risk predictor and a model report.

## Data

| Item | Detail |
|---|---|
| Source | DHS Program, Children's Recode (Stata `.dta`) |
| Used here | DHS model dataset (free, no registration, practice only) |
| Planned | Pakistan DHS 2017-18 (registration and approval required at dhsprogram.com) |
| Unit | One row per child |
| Population | Children alive at interview, aged 12 to 23 months (age = `v008 - b3`) |
| Size after filtering | 1,020 children |

**The DHS terms do not allow redistributing the survey files.** No raw DHS file is included in this repository. To reproduce, download the data yourself (see below).

### Target definition
Vaccine variables `h2` (BCG), `h3`, `h5`, `h7` (DPT 1-3), `h4`, `h6`, `h8` (Polio 1-3) and `h9` (Measles). A value of 1, 2 or 3 (card or mother's report) counts as vaccinated. Values of 0, 8 ("don't know") and missing count as **not vaccinated**, following the DHS convention. In the practice data, 68.4% of children are basic vaccinated and 31.6% are not.

### Features

| Variable | Meaning |
|---|---|
| `v025` | Urban / rural |
| `v024` | Region |
| `v190` | Wealth index (1 poorest to 5 richest) |
| `v106` | Mother's education |
| `v012` | Mother's age |
| `bord` (capped at 5) | Birth order |
| `b4` | Child's sex |
| `v151` | Sex of household head |
| `v467d` | Distance-to-facility problem (check codebook for labels) |
| `m14` (grouped) | Antenatal visits: 0, 1-3, 4+, unknown |
| `m15` (grouped) | Delivery place: home, public facility, private facility, other/unknown |

Survey weights (`v005 / 1,000,000`) are used for the **coverage percentages in the dashboard**. They are not used for model training.

## Method

- **Preprocessing:** standardised numeric features, one-hot encoded categorical features, all inside a scikit-learn `Pipeline` so there is no leakage across cross-validation folds.
- **Imbalance:** the classes are moderately balanced (31.6% positive), so no oversampling was used. Models use `class_weight="balanced"`.
- **Evaluation:** ROC-AUC, PR-AUC, recall and precision, because accuracy hides failures on the minority class. Out-of-fold predictions are used for the confusion matrix and group-wise results.
- **Explainability:** SHAP (`LinearExplainer`) and odds ratios.

## Results (practice dataset only)

5-fold stratified cross-validation, mean values:

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Dummy (baseline) | 0.500 | 0.316 | 0.000 | 0.000 | 0.000 |
| **Logistic Regression** | **0.642** | **0.480** | **0.596** | 0.433 | 0.499 |
| Random Forest | 0.626 | 0.422 | 0.494 | 0.408 | 0.446 |

Logistic Regression (out-of-fold): ROC-AUC 0.637, PR-AUC 0.460, recall 0.596, precision 0.428.

Confusion matrix (rows = actual, columns = predicted):

| | Predicted vaccinated | Predicted missed |
|---|---|---|
| **Actually vaccinated** | 441 | 257 |
| **Actually missed** | 130 | 192 |

Performance by group:

| Group | Children | AUC | Recall |
|---|---|---|---|
| Urban | 322 | 0.583 | 0.535 |
| Rural | 698 | 0.661 | 0.624 |
| Wealth 1 (poorest) | 283 | 0.538 | 0.451 |
| Wealth 2 | 216 | 0.701 | 0.653 |
| Wealth 3 | 202 | 0.625 | 0.573 |
| Wealth 4 | 215 | 0.672 | 0.662 |
| Wealth 5 (richest) | 104 | 0.638 | 0.697 |

### What these results do and do not show
- The model is better than the baseline, but the signal is **weak** (ROC-AUC around 0.64).
- Performance is **uneven across groups**, which is why group-wise evaluation is part of the pipeline. With 100 to 300 children per group, these differences may be noise.
- The strongest, best-supported associations are delivery at a **public facility** and **4 or more antenatal visits**, both linked to lower risk. Coefficients for very small groups (for example "other/unknown" delivery place, 6 children) are not reliable and are flagged in the dashboard.
- Some patterns (for example the direction of the wealth effect) are artefacts of the synthetic data and should not be interpreted.

## Dashboard

```
streamlit run app.py
```

| Tab | Content |
|---|---|
| Overview | Filters (area, region, wealth, mother's education), KPI cards, weighted coverage charts |
| Drivers | SHAP summary, coverage by antenatal visits and delivery place, coefficient table with small-group warnings |
| Risk Predictor | Enter a child's background and get a relative risk score, percentile and the main factors |
| Model | Model comparison, confusion matrix, group-wise performance, limitations |

The risk score comes from a class-weighted model, so it is **not a literal probability**. The dashboard shows a percentile against the dataset to make it interpretable.

*(Add a screenshot here: `docs/dashboard.png`)*

## Repository structure

```
.
├── vaxgap.py              # data cleaning, target, models, evaluation, SHAP, saves outputs
├── app.py                 # Streamlit dashboard
├── requirements.txt
├── vaxgap_model.joblib    # trained Logistic Regression pipeline (practice data)
├── vaxgap_clean.csv       # cleaned modelling table (practice data only)
├── shap_summary.png       # SHAP summary plot
└── README.md
```

## How to reproduce

```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv venv
venv\Scripts\activate          # Windows (use: source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```

1. Download the Children's Recode (Stata) model dataset from the DHS Program "Model Datasets" page (no registration needed), or the Pakistan DHS 2017-18 Children's Recode if you have approved access.
2. Put the `.dta` file in the project folder and set its file name in `vaxgap.py`.
3. Run `python vaxgap.py`. This prints the results and saves the model, the clean table and the SHAP plot.
4. Run `streamlit run app.py`.

If you switch to the Pakistan data, set `DATA_IS_PRACTICE = False` in `app.py`, update the region and distance labels from the DHS codebook, and update the numbers on the Model tab.

## Limitations

- Survey data only. The model has **never been tested in a real vaccination campaign**.
- Results show association, not cause.
- Missing and "don't know" vaccine answers are treated as not vaccinated.
- Small dataset, so confidence intervals are wide and some groups are too small to interpret.
- Model performance differs between groups and must be checked before any real-world use.
- This is a research demonstration, not a clinical or programme-planning tool.

## Next steps

- Run the full pipeline on the Pakistan DHS 2017-18 data and report real results.
- Add confidence intervals and calibration.
- Try gradient boosting and compare against the logistic baseline.
- Scale the data pipeline with PySpark to study behaviour on larger datasets.

## Data and licence

Code: choose a licence (for example MIT) and add a `LICENSE` file.
Data: DHS data is the property of The DHS Program (ICF) and is subject to its terms of use. Cite the survey when publishing results.

## Author

Your Name, Software Engineering graduate (Mehran University of Engineering and Technology). Interested in machine learning for public health.
[LinkedIn] | [Email]
