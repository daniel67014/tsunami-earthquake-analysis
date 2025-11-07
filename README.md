## Tsunami–Earthquake Analysis

This project builds a recall-first classifier to flag tsunami-related earthquakes from USGS-style seismic records. It includes a reproducible ETL and feature engineering pipeline (with interaction mining), model training and threshold tuning, and persisted artifacts ready for inference. A lightweight dashboard (Streamlit) is planned to make predictions and visualize model behavior.

### Repository structure

```
data/
	raw/                      # Source CSV (earthquake_data_tsunami.csv)
	processed/                # Imputed base features
	interaction/              # 2-way interaction exports (imputed/raw)
jupyter_notebooks/
	01_etl_extract_raw.ipynb
	02_01_etl_transform.ipynb
	02_02_etl_feature_engineering.ipynb
	03_model_training_and_prediction.ipynb
	models/
		rf_imputed_selected.pkl
		model_meta.json
		threshold.txt
Procfile
requirements.txt
setup.sh
```

## What’s implemented

- ETL and cleaning (01, 02_01)
	- Handles missingness; produces an imputed dataset.
	- Reproducibility controls (fixed seeds; documented where upstream algorithms like KMeans can still vary by permutation).
- Feature engineering (02_02)
	- 2-way interaction mining and forward selection.
	- Final selected feature sets:
		- Imputed (8): dmin, Year, cdi, dmin:Year, gap, sig, magnitude, depth
		- Raw (6): Year, nst, sig, magnitude, Year:magnitude, depth
- Modeling and evaluation (03)
	- RandomForestClassifier (n_estimators=100, max_depth=8, class_weight=balanced, random_state=42).
	- Recall-first policy with precision–recall threshold sweep.
	- Current choice: imputed-selected model with threshold = 0.0 → recall 1.00, precision ~0.389 (meets recall ≥ 0.88, precision ≥ 0.30).
	- Utilities: threshold sweep summary, confusion matrix at chosen threshold, model persistence.
- Artifacts (persisted)
	- `jupyter_notebooks/models/rf_imputed_selected.pkl`
	- `jupyter_notebooks/models/model_meta.json` (features, threshold, versions, seed)
	- `jupyter_notebooks/models/threshold.txt`

## Reproducibility

- Fixed random_state=42 across steps; KMeans and other stochastic components have explicit seeds and documented `n_init`.
- Notebooks record selected features and model hyperparameters.
- `model_meta.json` carries the canonical feature list and decision threshold used at inference time.

## How to run locally

1. Create/activate a Python 3.12 virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run notebooks in order: 01 → 02_01 → 02_02 → 03.
4. Notebook 03 will train models, perform threshold selection, and persist artifacts under `jupyter_notebooks/models/`.

Tip: Re-running cells changes notebook outputs and will show as modified in source control; commit when you want to capture results.

## Model details

- Target: tsunami-related earthquake flag.
- Best current model: RandomForest on the imputed-selected 8-feature set.
- Decision threshold: 0.0 (recall-first). Candidates with recall 1.0 and higher precision also exist (e.g., small >0 thresholds); this can be tuned depending on tolerance for false positives.

## Planned dashboard (Streamlit)

Goals:
- Load the persisted model and threshold.
- Accept manual inputs or a small CSV for batch scoring (enforcing the 8-feature schema from `model_meta.json`).
- Display key outputs: predicted probability, decision (using threshold), feature importances, confusion matrix snapshot, and a precision–recall curve.

MVP pages:
- Predict: form inputs + single prediction result.
- Batch: upload CSV → predictions + download.
- Explain: feature importance bar chart and notes on limitations.

Deployment options:
- Streamlit Community Cloud (simple) or Heroku (Procfile present; add `streamlit run app.py` entry). A `setup.sh` exists; adapt as needed.

## Roadmap

- [ ] Add Streamlit app (`app.py`) wired to persisted artifacts.
- [ ] Add schema validator for inference (strict column presence and dtypes).
- [ ] Document a recommended operating threshold (consider precision-improving alternative at same recall).
- [ ] Add a minimal model card (assumptions, limitations, intended use).
- [ ] Optional: persist a raw-selected model for comparison.

## Acknowledgments

- Built as part of a Code Institute Data Analytics capstone.
- Thanks to USGS-style earthquake data sources and open-source libraries (pandas, scikit-learn, xgboost, seaborn, matplotlib, joblib).

