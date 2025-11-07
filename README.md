## Tsunami–Earthquake Analysis

This project builds a recall-first classifier to flag tsunami-related earthquakes from USGS-style seismic records. It includes a reproducible ETL and feature engineering pipeline (with interaction mining), model training and threshold tuning, and persisted artifacts ready for inference. A lightweight dashboard (Streamlit) provides Predict, Batch, Explain, Stats, and About pages. See `docs/pass_criteria.md` for the assessment mapping.

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

## Code Quality & Optimization

This project demonstrates several best practices and improvements:

### Modular Design
- **Reusable utilities**: `ensure_schema()`, `parse_interactions()`, and `compute_interaction_columns()` encapsulate schema validation and feature engineering logic, reducing duplication across prediction/batch workflows.
- **Separation of concerns**: Model loading, threshold reading, and inference separated into focused functions.

### Performance Optimization
- **Caching**: Streamlit's `@st.cache_resource` avoids repeated model deserialization; `@st.cache_data` caches metadata loading.
- **Vectorization**: Interaction terms computed via NumPy broadcasting (no Python loops), improving throughput for batch scoring.
- **Lazy evaluation**: Dashboard pages load data only when accessed.

### Error Handling & Validation
- **Descriptive errors**: `ValueError` messages specify missing features by name, aiding debugging.
- **Schema enforcement**: `ensure_schema()` prevents runtime type errors by coercing numeric columns and validating presence.
- **Graceful fallbacks**: Default threshold (0.0) used if `threshold.txt` is unreadable.

### Documentation
- **Docstrings**: Key functions include Google-style docstrings with Args/Returns/Raises sections (see `streamlit_app.py`).
- **Inline rationale**: Comments explain recall-first policy, interaction parsing logic, and reproducibility constraints.
- **Version pinning**: `requirements.txt` specifies exact versions to ensure consistent environments.

### Refactoring Examples
- **Before**: Inline CSV column checks scattered across cells → **After**: Centralized `ensure_schema()` utility.
- **Before**: Hardcoded file paths → **After**: `Path().resolve()` for cross-platform portability.
- **Before**: Manual feature list synchronization → **After**: `model_meta.json` as single source of truth.

These improvements support maintainability, reproducibility, and performance—critical for transitioning from prototype to production-ready analytics.

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

## Dashboard (Streamlit)

Implemented pages:
- Predict: form inputs + single prediction result.
- Batch: upload CSV → predictions + download.
- Explain: feature importance bar chart and meta info.
- Stats: descriptive statistics (mean, median, std, variance) & distribution with normal overlay.
- About: non-technical summary, ethics, limitations, learning outcomes linkage.

Upcoming enhancements:
- Precision–recall curve visual in app (currently in Notebook 03).
- Adjustable threshold slider to explore precision/recall trade-offs.

### Running the app locally

```bash
streamlit run app/streamlit_app.py
```

Or (Heroku / Procfile expects `app.py`): create a thin `app.py` that imports and runs `streamlit_app.py`, or adjust the `Procfile` to point directly to `app/streamlit_app.py`.

### Heroku deployment note

Current `Procfile` runs: `streamlit run app.py`. Update it to:

```
web: sh setup.sh && streamlit run app/streamlit_app.py
```

Then push to the Heroku-connected branch.

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

## Data Sources & Licensing

- **Dataset**: Earthquake records with tsunami flag (USGS-style format). Public domain seismic data aggregated for educational use.
- **Licensing**: This project is for academic demonstration. Source data follows public domain or open data policies; always verify current licensing for production use.
- **Societal Impact**: Tsunami early-warning systems save lives. This model is exploratory only; operational systems require rigorous validation, multi-sensor fusion, and regulatory approval.

## AI Tool Usage

- **GitHub Copilot**: Assisted with boilerplate code generation, interaction term mining logic, and threshold sweep utilities. Suggestions were reviewed and adapted to project-specific requirements.
- **Impact**: Accelerated feature engineering experimentation and improved code documentation consistency. Human oversight ensured alignment with recall-first policy and reproducibility standards.

## Reflection & Learning

### Skills Gained
- End-to-end ML pipeline: from raw data extraction through feature engineering, model selection, threshold tuning, and artifact persistence.
- Recall-first evaluation strategy and precision–recall trade-off analysis.
- Interactive dashboard design (Streamlit) for non-technical stakeholders.
- Version control best practices and structured project documentation.

### Challenges Encountered
- Feature mismatches between engineered CSVs and model schemas required careful validation utilities.
- Class imbalance necessitated threshold sweep rather than default 0.5 cutoff.
- Balancing reproducibility (fixed seeds) with stochastic upstream algorithms (KMeans).

### Future Learning Plan
- Explore calibrated probability models (Platt scaling, isotonic regression) for improved confidence estimates.
- Investigate cost-sensitive learning and dynamic threshold adjustment based on operational context.
- Study multi-modal fusion (seismic + oceanographic sensors) for real-time early-warning architectures.
- Deepen understanding of MLOps: model monitoring, drift detection, and automated retraining pipelines.


