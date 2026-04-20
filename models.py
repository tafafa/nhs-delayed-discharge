"""
models.py
---------
Model training, evaluation, and comparison for NHS Delayed Discharge prediction.

Models:
    - Linear Regression (baseline)
    - Random Forest Regressor
    - XGBoost Regressor
    - Gradient Boosting Regressor

Evaluation:
    - MAE, RMSE, R² on held-out test set
    - 5-Fold Cross-Validated R²
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "prevmonthdelay",
    "healthsocialrate",
    "patientfamilyrate",
    "awirate",
    "nonawirate",
    "elderlyrate",
    "carehomecapacityrate",
    "carehomeplaces",
    "month",
    "year",
    "hb_encoded"
]

TARGET_COLUMN = "totaldelayedbeddays"


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def encode_health_board(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Label-encode the Health Board identifier column."""
    le = LabelEncoder()
    df = df.copy()
    df["hb_encoded"] = le.fit_transform(df["hbt"])
    return df, le


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (target) from the modelling DataFrame."""
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models() -> dict:
    """Return a dictionary of named, untrained model instances."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict of MAE, RMSE, and R² for a set of predictions."""
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
    cv_folds: int = 5
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Full training and evaluation pipeline.

    Returns:
        trained_models  - dict of {model_name: fitted model}
        results         - dict of {model_name: {MAE, RMSE, R2, CV_R2_mean, CV_R2_std}}
        comparison_df   - DataFrame summarising all results
    """
    df, _ = encode_health_board(df)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    models = build_models()
    trained_models = {}
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_model(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="r2")
        metrics["CV_R2_mean"] = round(cv_scores.mean(), 4)
        metrics["CV_R2_std"] = round(cv_scores.std(), 4)

        trained_models[name] = model
        results[name] = metrics
        print(f"  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}  "
              f"R²={metrics['R2']}  CV R²={metrics['CV_R2_mean']} "
              f"(±{metrics['CV_R2_std']})")

    comparison_df = pd.DataFrame(results).T
    return trained_models, results, comparison_df


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    save_path: str = None
) -> None:
    """Scatter plot of actual vs predicted values with a perfect-prediction line."""
    y_pred = model.predict(X_test)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.4)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    plt.xlabel("Actual Delayed Bed Days")
    plt.ylabel("Predicted Delayed Bed Days")
    plt.title(f"{model_name} — Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_residuals(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    save_path: str = None
) -> None:
    """Residual plot (predicted vs residuals) with a zero-error reference line."""
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.4)
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted Delayed Bed Days")
    plt.ylabel("Residuals")
    plt.title(f"{model_name} — Residual Plot")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_cv_r2_comparison(
    results: dict,
    save_path: str = None
) -> None:
    """Bar chart comparing 5-fold CV R² scores across all models."""
    names = list(results.keys())
    cv_means = [results[n]["CV_R2_mean"] for n in names]
    cv_stds = [results[n]["CV_R2_std"] for n in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, cv_means, yerr=cv_stds, capsize=5,
                   color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    plt.ylabel("CV R²")
    plt.title("5-Fold Cross-Validated R² by Model")
    plt.ylim(0, 1.05)
    for bar, val in zip(bars, cv_means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
