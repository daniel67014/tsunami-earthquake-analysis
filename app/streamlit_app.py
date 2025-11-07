import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import joblib


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


def main():
    st.set_page_config(
        page_title="Tsunami–Earthquake Classifier", layout="wide"
    )

    st.sidebar.title("Tsunami–Earthquake Analysis")
    page = st.sidebar.radio("Navigation", ["Predict", "Batch", "Explain"])

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
    else:
        page_explain(meta, model)


if __name__ == "__main__":
    main()
