from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, precision_score, roc_auc_score


def precision_at_k(y_true, y_score, k: float = 0.1) -> float:
    """Precision among the top k fraction or top k absolute ranked predictions."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if 0 < k < 1:
        cutoff = max(1, int(np.ceil(len(y_score) * k)))
    else:
        cutoff = int(k)

    cutoff = min(max(cutoff, 1), len(y_score))
    top_indices = np.argsort(y_score)[::-1][:cutoff]
    return float(np.mean(y_true[top_indices]))


def compute_classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
        "precision_at_1_pct": precision_at_k(y_true, y_score, 0.01),
        "precision_at_5_pct": precision_at_k(y_true, y_score, 0.05),
        "precision_at_10_pct": precision_at_k(y_true, y_score, 0.10),
        "precision_at_threshold_0_5": float(precision_score(y_true, y_pred, zero_division=0)),
        "mean_predicted_ctr": float(np.mean(y_score)),
        "observed_ctr": float(np.mean(y_true)),
    }


def calibration_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"predicted_ctr": prob_pred, "observed_ctr": prob_true})


def lift_chart_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    ranked = pd.DataFrame({"actual": y_true, "score": y_score}).sort_values("score", ascending=False)
    ranked["decile"] = pd.qcut(np.arange(len(ranked)), q=n_bins, labels=False) + 1
    baseline = ranked["actual"].mean()

    lift = (
        ranked.groupby("decile", as_index=False)
        .agg(
            users=("actual", "size"),
            clicks=("actual", "sum"),
            predicted_ctr=("score", "mean"),
            observed_ctr=("actual", "mean"),
        )
        .sort_values("decile")
    )
    lift["lift"] = lift["observed_ctr"] / baseline if baseline else 0.0
    return lift
