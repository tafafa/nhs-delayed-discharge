"""
explainability.py
-----------------
SHAP-based model explainability for NHS Delayed Discharge prediction.

Provides:
    - Global feature importance (bar chart)
    - SHAP summary (beeswarm) plot
    - Individual prediction explanation (waterfall plot)
"""

import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    """
    Compute SHAP values for a trained tree-based model.

    Uses TreeExplainer for Random Forest, XGBoost, and Gradient Boosting.
    Falls back to LinearExplainer for Linear Regression.

    Args:
        model: A fitted scikit-learn or XGBoost model.
        X: Feature DataFrame used to compute SHAP values.

    Returns:
        shap.Explanation object containing SHAP values and base values.
    """
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.LinearExplainer(model, X)

    shap_values = explainer(X)
    return shap_values


def plot_shap_bar(
    shap_values: shap.Explanation,
    model_name: str,
    max_display: int = 10,
    save_path: str = None
) -> None:
    """
    Plot a SHAP global feature importance bar chart.

    Args:
        shap_values: SHAP Explanation object.
        model_name: Name of the model (used in plot title).
        max_display: Number of top features to display.
        save_path: Optional file path to save the figure.
    """
    plt.figure()
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_summary(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    model_name: str,
    max_display: int = 10,
    save_path: str = None
) -> None:
    """
    Plot a SHAP beeswarm summary plot showing direction and magnitude of feature effects.

    Args:
        shap_values: SHAP Explanation object.
        X: Feature DataFrame (needed for feature values).
        model_name: Name of the model (used in plot title).
        max_display: Number of top features to display.
        save_path: Optional file path to save the figure.
    """
    plt.figure()
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title(f"SHAP Summary Plot — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_waterfall(
    shap_values: shap.Explanation,
    instance_index: int = 0,
    model_name: str = "",
    save_path: str = None
) -> None:
    """
    Plot a SHAP waterfall chart explaining a single prediction.

    Args:
        shap_values: SHAP Explanation object.
        instance_index: Index of the instance to explain (default: 0).
        model_name: Name of the model (used in plot title).
        save_path: Optional file path to save the figure.
    """
    plt.figure()
    shap.plots.waterfall(shap_values[instance_index], show=False)
    plt.title(f"SHAP Waterfall — {model_name} (Instance {instance_index})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def get_mean_shap_importances(
    shap_values: shap.Explanation,
    feature_names: list
) -> pd.DataFrame:
    """
    Return a DataFrame of mean absolute SHAP values, sorted descending.

    Args:
        shap_values: SHAP Explanation object.
        feature_names: List of feature column names.

    Returns:
        DataFrame with columns ['feature', 'mean_abs_shap'].
    """
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance_df
