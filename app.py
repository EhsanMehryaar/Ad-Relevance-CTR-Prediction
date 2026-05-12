from __future__ import annotations

import json
import inspect
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import sklearn
import streamlit as st

from src.config import CATEGORICAL_FEATURES, INTEGER_FEATURES, MODELS_DIR
from src.features import add_engineered_features


st.set_page_config(page_title="Ad Relevance & CTR Prediction", layout="wide")


def artifact_path(filename: str) -> Path:
    return MODELS_DIR / filename


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None, None
    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, exc


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_prediction_frame() -> pd.DataFrame:
    st.sidebar.header("Prediction Inputs")
    st.sidebar.caption("Criteo fields are anonymized, so inputs are grouped as user, context, and ad signals.")

    values: dict[str, object] = {}
    with st.sidebar.expander("User numeric context", expanded=True):
        for column in INTEGER_FEATURES[:4]:
            values[column] = st.number_input(column, value=0.0, step=1.0)

    with st.sidebar.expander("Video/context numeric signals", expanded=False):
        for column in INTEGER_FEATURES[4:8]:
            values[column] = st.number_input(column, value=0.0, step=1.0)

    with st.sidebar.expander("Ad numeric signals", expanded=False):
        for column in INTEGER_FEATURES[8:]:
            values[column] = st.number_input(column, value=0.0, step=1.0)

    with st.sidebar.expander("Categorical IDs", expanded=False):
        for column in CATEGORICAL_FEATURES:
            values[column] = st.text_input(column, value="__missing__")

    return add_engineered_features(pd.DataFrame([values]))


def metric_card(label: str, value: object) -> None:
    if isinstance(value, float):
        st.metric(label, f"{value:.4f}")
    else:
        st.metric(label, value)


def plot_chart(fig) -> None:
    plotly_params = inspect.signature(st.plotly_chart).parameters
    if "width" in plotly_params:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def show_dataframe(df: pd.DataFrame) -> None:
    dataframe_params = inspect.signature(st.dataframe).parameters
    if "width" in dataframe_params and "use_container_width" not in dataframe_params:
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


st.title("Ad Relevance & CTR Prediction System")

model, model_error = load_model(artifact_path("model.joblib"))
metrics = load_metrics(artifact_path("metrics.json"))
metadata = load_metadata(artifact_path("model_metadata.json"))
feature_importance = load_csv(artifact_path("feature_importance.csv"))
calibration = load_csv(artifact_path("calibration_curve.csv"))
lift = load_csv(artifact_path("lift_chart.csv"))

if model_error is not None:
    trained_sklearn = metadata.get("sklearn_version", "unknown")
    st.error(
        "The saved model artifact could not be loaded. "
        f"It was trained with scikit-learn {trained_sklearn}, while this app is running "
        f"scikit-learn {sklearn.__version__}. Retrain the model in the same environment "
        "you use to run Streamlit."
    )
    st.code("python -m src.train --data-path data/raw/criteo_train.txt --model xgboost --sample-size 200000")
    with st.expander("Technical error"):
        st.exception(model_error)
elif model is None:
    st.warning("No trained model found. Train one first with `python -m src.train --data-path data/raw/criteo_train.txt`.")
else:
    prediction_frame = build_prediction_frame()
    predicted_ctr = float(model.predict_proba(prediction_frame)[:, 1][0])
    st.sidebar.metric("Predicted CTR", f"{predicted_ctr:.2%}")

if metrics:
    st.subheader("Model Performance")
    cols = st.columns(4)
    key_metrics = [
        ("AUC-ROC", metrics.get("auc_roc")),
        ("Log Loss", metrics.get("log_loss")),
        ("Precision@10%", metrics.get("precision_at_10_pct")),
        ("Observed CTR", metrics.get("observed_ctr")),
    ]
    for col, (label, value) in zip(cols, key_metrics):
        with col:
            metric_card(label, value)

    st.caption(
        f"Model: {metrics.get('model_key', 'unknown')} | "
        f"Train rows: {metrics.get('train_rows', 'n/a')} | Test rows: {metrics.get('test_rows', 'n/a')}"
    )
    if metadata:
        st.caption(
            f"Artifact environment: Python {metadata.get('python_version', 'unknown')} | "
            f"scikit-learn {metadata.get('sklearn_version', 'unknown')}"
        )
else:
    st.info("Metrics will appear here after training.")

chart_cols = st.columns(2)

with chart_cols[0]:
    st.subheader("Calibration Curve")
    if calibration.empty:
        st.info("Calibration artifact not found.")
    else:
        fig = px.line(
            calibration,
            x="predicted_ctr",
            y="observed_ctr",
            markers=True,
            labels={"predicted_ctr": "Predicted CTR", "observed_ctr": "Observed CTR"},
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "gray"})
        plot_chart(fig)

with chart_cols[1]:
    st.subheader("Lift Chart")
    if lift.empty:
        st.info("Lift chart artifact not found.")
    else:
        fig = px.bar(
            lift,
            x="decile",
            y="lift",
            hover_data=["users", "clicks", "observed_ctr", "predicted_ctr"],
            labels={"decile": "Ranked score decile", "lift": "Lift vs baseline CTR"},
        )
        plot_chart(fig)

st.subheader("Feature Importance")
if feature_importance.empty:
    st.info("Feature importance is available for logistic regression, random forest, XGBoost, and LightGBM models.")
else:
    top_features = feature_importance.head(25).sort_values("importance")
    fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    plot_chart(fig)
    show_dataframe(feature_importance)
