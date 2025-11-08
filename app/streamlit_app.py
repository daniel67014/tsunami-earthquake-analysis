import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- Paths & loading ----------
MODELS_DIR = (
    Path(__file__).resolve().parent.parent / "jupyter_notebooks" / "models"
).resolve()
MODEL_PATH = MODELS_DIR / "rf_imputed_selected.pkl"
META_PATH = MODELS_DIR / "model_meta.json"
THRESHOLD_PATH = MODELS_DIR / "threshold.txt"


def _read_threshold() -> float:
    try:
        return float(THRESHOLD_PATH.read_text().strip())
    except Exception:
        # Safe default for recall-first; can be adjusted via UI later
        return 0.0


@st.cache_data(show_spinner=False)
def load_meta() -> Dict:
    with META_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load(MODEL_PATH)
    return model


def parse_interactions(
    features: List[str],
) -> Tuple[List[str], List[Tuple[str, ...]]]:
    """
    Parse feature list to separate base features from interaction terms.

    Interaction terms are identified by ':' separator (e.g., 'dmin:Year').
    Maintains stable ordering: non-interaction features first (in original
    order), then any additional base features, then interactions.

    Args:
        features: List of feature names, may include interactions.

    Returns:
        Tuple of (ordered_base_features, interaction_tuples).
        Example: (['dmin', 'Year'], [('dmin', 'Year')])
    """
    base = set()
    interactions = []
    for f in features:
        if ":" in f:
            parts = tuple(p.strip() for p in f.split(":"))
            interactions.append(parts)
            for p in parts:
                base.add(p)
        else:
            base.add(f)
    # Keep order stable: first add all non-interaction features in given order,
    # then interactions
    ordered_base = []
    seen = set()
    for f in features:
        if ":" not in f and f not in seen:
            ordered_base.append(f)
            seen.add(f)
    for b in sorted(base - set(ordered_base)):
        ordered_base.append(b)
    return ordered_base, interactions


def compute_interaction_columns(
    df: pd.DataFrame, interactions: List[Tuple[str, ...]]
) -> pd.DataFrame:
    """
    Compute interaction (product) terms from base columns.

    Creates new columns by multiplying base features together.
    Example: ('dmin', 'Year') → column 'dmin:Year' = dmin * Year.

    Args:
        df: DataFrame containing base features.
        interactions: List of tuples, each tuple defines features to multiply.

    Returns:
        DataFrame with interaction columns added.

    Raises:
        ValueError: If any base column required for interaction is missing.

    Note:
        Uses NumPy broadcasting for vectorized computation (no loops).
    """
    df = df.copy()
    for parts in interactions:
        name = ":".join(parts)
        # If interaction already exists, skip recompute
        if name in df.columns:
            continue
        missing = [p for p in parts if p not in df.columns]
        if missing:
            # Can't compute this interaction
            raise ValueError(
                f"Missing base columns for interaction '{name}': {missing}"
            )
        col = np.ones(len(df))
        for p in parts:
            col = col * pd.to_numeric(df[p], errors="coerce").astype(float)
        df[name] = col
    return df


def ensure_schema(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    """
    Validate and align DataFrame to model's required schema.

    Ensures all required features (including interactions) are present and
    correctly typed. Computes interaction columns if needed from base features.

    Args:
        df: Input DataFrame with raw or partially processed features.
        required: List of required feature names (may include interactions).

    Returns:
        DataFrame with only required columns, numeric typed, in correct order.

    Raises:
        ValueError: If required features are missing after interaction
            computation.

    Example:
        >>> required = ['dmin', 'Year', 'dmin:Year']
        >>> df_aligned = ensure_schema(df, required)
    """
    # Compute interactions if needed
    _, inters = parse_interactions(required)
    df2 = compute_interaction_columns(df, inters) if inters else df.copy()

    # Ensure all required present
    missing = [c for c in required if c not in df2.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Coerce numeric types; RF tolerates order but we will align explicitly
    for c in required:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").astype(float)

    return df2[required]


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)[:, 1]
    return proba


def threshold_decision(proba: np.ndarray, thr: float) -> np.ndarray:
    return (proba >= thr).astype(int)


# ---------- UI ----------

def page_predict(meta: Dict, model) -> None:
    st.header("Single prediction")
    required: List[str] = meta.get("features", [])
    base, inters = parse_interactions(required)

    st.caption(
        "Enter base features; interaction columns will be computed "
        "automatically."
    )

    cols = st.columns(min(3, len(base)) or 1)
    inputs = {}
    for i, feat in enumerate(base):
        with cols[i % len(cols)]:
            default = 0.0
            if feat.lower() == "year":
                default = 2015
            val = st.number_input(feat, value=float(default))
            inputs[feat] = val

    # Build DataFrame and compute interactions
    df = pd.DataFrame([inputs])
    try:
        X = ensure_schema(df, required)
    except Exception as e:
        st.error(str(e))
        return

    thr = _read_threshold()
    proba = predict_proba(model, X)[0]
    pred = int(proba >= thr)

    st.subheader("Result")
    st.metric(label="Predicted probability (tsunami)", value=f"{proba:.3f}")
    st.metric(label="Decision", value="Positive" if pred == 1 else "Negative")
    st.caption(
        f"Threshold = {thr:.3f} | Model = {meta.get('model', '')} | "
        f"random_state = {meta.get('random_state')}"
    )


def page_batch(meta: Dict, model) -> None:
    st.header("Batch scoring")
    required: List[str] = meta.get("features", [])
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.info(
            "Provide a CSV containing base features. Interaction columns will "
            "be computed if base columns exist."
        )
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    try:
        X = ensure_schema(df, required)
    except Exception as e:
        st.error(str(e))
        return

    thr = _read_threshold()
    proba = predict_proba(model, X)
    pred = threshold_decision(proba, thr)

    out = df.copy()
    out["proba_tsunami"] = proba
    out["decision"] = pred

    st.subheader("Preview")
    st.dataframe(out.head(20))

    st.download_button(
        "Download predictions CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )


def page_explain(meta: Dict, model) -> None:
    st.header("Explainability")
    feats: List[str] = meta.get("features", [])
    thr = _read_threshold()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df_imp = (
            pd.DataFrame({"feature": feats, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        st.subheader("Feature importances (RandomForest)")
        st.bar_chart(df_imp.set_index("feature"))
        with st.expander("Show table"):
            st.dataframe(df_imp)
    else:
        st.info("Model does not expose feature_importances_.")

    st.caption(
        f"Threshold = {thr:.3f} | Features = {len(feats)} | "
        f"sklearn = {meta.get('sklearn_version')} | "
        f"xgboost = {meta.get('xgboost_version')}"
    )


def page_stats() -> None:
    st.header("Dataset Statistics")
    raw_path = (
        Path(__file__).resolve().parent.parent
        / "data" / "raw" / "earthquake_data_tsunami.csv"
    )
    if not raw_path.exists():
        st.error(f"Raw data file not found: {raw_path}")
        return
    df = pd.read_csv(raw_path)
    numeric_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        st.warning("No numeric columns detected.")
        return

    st.subheader("Summary (mean, median, std, variance)")
    summary = df[numeric_cols].describe().T
    summary["variance"] = df[numeric_cols].var().values
    st.dataframe(summary[["mean", "50%", "std", "variance", "min", "max"]])

    # Choose magnitude-like column for histogram
    col_candidates = [
        c for c in numeric_cols if c.lower() in ["mag", "magnitude"]
    ]
    col = col_candidates[0] if col_candidates else numeric_cols[0]

    st.subheader(f"Distribution: {col}")
    data = df[col].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        data,
        bins=30,
        kde=True,
        stat="density",
        ax=ax,
        color="#4e79a7",
    )
    mu, sigma = np.nanmean(data), np.nanstd(data)
    xs = np.linspace(data.min(), data.max(), 200)
    normal_pdf = (
        1.0
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
    )
    ax.plot(
        xs,
        normal_pdf,
        color="#e15759",
        lw=2,
        label=f"Normal fit (μ={mu:.2f}, σ={sigma:.2f})",
    )
    ax.legend()
    ax.set_title(f"Distribution of {col} with normal overlay")
    st.pyplot(fig)

    st.caption(
        "Shows LO 1.1 & 1.2: descriptive statistics & distribution analysis."
    )


def page_about() -> None:
    st.header("About & Ethics")
    
    # AI-generated executive summary for non-technical audiences
    st.subheader("Executive Summary (AI-Generated)")
    st.markdown(
        """
        **What does this model do?**
        This tool analyzes earthquake data to predict whether a quake might
        trigger a tsunami. It uses patterns in seismic measurements—like
        location, depth, and magnitude—to flag high-risk events.

        **Why does it matter?**
        Tsunamis can devastate coastal communities with little warning. By
        quickly identifying tsunami-related earthquakes, this model could
        support early-alert systems, giving people precious extra minutes
        to evacuate.

        **How reliable is it?**
        The model is tuned to catch nearly all tsunami-related quakes
        (100% recall), but it also raises false alarms about 61% of the time.
        This trade-off prioritizes safety: better to warn unnecessarily than
        miss a real threat. Always corroborate predictions with authoritative
        tsunami warning centers.

        **Who should use it?**
        Educational demonstrations, risk assessment planning, and
        research prototyping. NOT for operational emergency response
        without rigorous validation.
        """
    )

    st.subheader("Data Sources & Legal Considerations")
    st.markdown(
        """
        - **Data**: Public domain seismic records (USGS-style format)
        aggregated for academic use.
        - **Licensing**: This project is educational. Verify current data
        licenses before any commercial or operational deployment.
        - **Privacy**: No personal data involved; all records are
        aggregated geophysical measurements.
        """
    )

    st.subheader("Intended Use")
    st.markdown(
        """
        Exploratory triage and educational demonstration. Not for
        operational early-warning decisions.
        """
    )

    st.subheader("Ethical & Social Considerations")
    st.markdown(
        """
        - **Recall-first trade-off**: Minimises missed tsunami-related
        events (false negatives) while accepting more false positives.
        - **Bias & Data Quality**: Regional sensor coverage and reporting
        standards may bias feature distributions.
        - **Transparency**: We expose model threshold, recall, and
        precision to avoid overstating reliability.
        - **Misuse Risks**: Decisions solely based on this model could
        misallocate resources; always corroborate with authoritative
        sources.
        - **Societal Impact**: Effective early-warning systems save lives.
        This prototype demonstrates feasibility but requires operational
        validation, regulatory approval, and integration with existing
        tsunami alert infrastructure.
        """
    )

    st.subheader("Limitations & Alternatives")
    st.markdown(
        """
        - Potential improvement via calibrated logistic regression or
        cost-sensitive learning.
        - Continuous monitoring for drift recommended if deployed.
        - Multi-sensor fusion (oceanographic buoys, GPS displacement)
        would improve accuracy.
        """
    )

    st.subheader("Learning Outcomes Covered")
    st.caption(
        "Ethics & governance (6.1, 6.2), AI storytelling (4.2), "
        "communication (8.1, 8.2), domain context (9.x), "
        "reflection (10.2, 11.2)."
    )


def page_notebooks():
    """
    Display Jupyter notebooks with navigation and embedding options.

    Provides links to view notebooks on GitHub or nbviewer, with an option
    to render them inline using Streamlit's iframe component.
    """
    st.title("📓 Jupyter Notebooks")

    st.markdown(
        """
    The analysis is documented across four Jupyter notebooks that should
    be run in sequence. Each notebook includes detailed markdown commentary
    explaining methodology, decisions, and limitations.
    """
    )

    notebooks = [
        {
            "number": "1",
            "title": "Data Collection",
            "file": "01_etl_extract_raw.ipynb",
            "description": (
                "Extract raw data from Kaggle, perform initial inspection, "
                "and apply foundational statistical concepts."
            ),
        },
        {
            "number": "2",
            "title": "Data Cleaning and Imputation",
            "file": "02_etl_transform.ipynb",
            "description": (
                "Handle missing values, benchmark imputation strategies, "
                "and produce cleaned dataset."
            ),
        },
        {
            "number": "3",
            "title": "Feature Engineering",
            "file": "03_etl_feature_engineering.ipynb",
            "description": (
                "Generate interaction terms, perform feature selection, "
                "and export enhanced datasets."
            ),
        },
        {
            "number": "4",
            "title": "Model Training and Evaluation",
            "file": "04_model_training_and_prediction.ipynb",
            "description": (
                "Train models, tune threshold for recall optimization, "
                "and persist artifacts."
            ),
        },
    ]

    st.markdown("---")

    # Display each notebook with options to view
    for nb in notebooks:
        st.subheader(f"Notebook {nb['number']}: {nb['title']}")
        st.markdown(nb["description"])

        col1, col2, col3 = st.columns([2, 2, 3])

        with col1:
            github_base = (
                "https://github.com/daniel67014/"
                "tsunami-earthquake-analysis/blob/main/jupyter_notebooks/"
            )
            github_url = f"{github_base}{nb['file']}"
            st.markdown(f"[📖 View on GitHub]({github_url})")

        with col2:
            nbviewer_base = (
                "https://nbviewer.org/github/daniel67014/"
                "tsunami-earthquake-analysis/blob/main/jupyter_notebooks/"
            )
            nbviewer_url = f"{nbviewer_base}{nb['file']}"
            st.markdown(f"[🔍 View in nbviewer]({nbviewer_url})")

        with col3:
            # Local file path option (only works when running locally)
            local_path = (
                Path(__file__).resolve().parent.parent
                / "jupyter_notebooks"
                / nb["file"]
            )
            if local_path.exists():
                st.caption(f"✓ Available locally: `{nb['file']}`")
            else:
                st.caption("⚠ Not found locally")

        st.markdown("---")

    st.info(
        """
    **Viewing Options:**
    - **GitHub**: Basic rendering with syntax highlighting
    - **nbviewer**: Enhanced rendering with better notebook display
    - **Local**: Clone the repository and run notebooks in Jupyter/VS Code
      for full interactivity
    """
    )

    st.markdown("### 🔗 Quick Links")
    repo_base = "https://github.com/daniel67014/tsunami-earthquake-analysis"
    st.markdown(f"- [📚 README Documentation]({repo_base}/blob/main/README.md)")
    st.markdown(
        f"- [📋 Assessment Criteria Mapping]"
        f"({repo_base}/blob/main/docs/pass_criteria.md)"
    )
    st.markdown(f"- [💻 GitHub Repository]({repo_base})")


def main():
    st.set_page_config(
        page_title="Tsunami–Earthquake Classifier", layout="wide"
    )

    st.sidebar.title("Tsunami–Earthquake Analysis")
    page = st.sidebar.radio(
        "Navigation",
        ["Predict", "Batch", "Explain", "Stats", "Notebooks", "About"],
    )

    try:
        meta = load_meta()
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load model artifacts from {MODELS_DIR}: {e}")
        st.stop()

    if page == "Predict":
        page_predict(meta, model)
    elif page == "Batch":
        page_batch(meta, model)
    elif page == "Explain":
        page_explain(meta, model)
    elif page == "Stats":
        page_stats()
    elif page == "Notebooks":
        page_notebooks()
    else:
        page_about()


if __name__ == "__main__":
    main()
