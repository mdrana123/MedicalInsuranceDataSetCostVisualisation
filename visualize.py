import matplotlib
matplotlib.use("Agg")  # safe for saving files / CI. Remove if you want GUI pop-ups.

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd


def _ensure_figs():
    Path("figures").mkdir(exist_ok=True)


def get_feature_names(pipeline):
    """Get transformed feature names from the preprocessor inside a pipeline."""
    pre = pipeline.named_steps["preprocess"]
    return pre.get_feature_names_out()  # e.g. ['cat__sex_male','cat__smoker_yes','num__age',...]


def feature_importance_df(pipeline):
    """
    Returns a DataFrame with 'feature' and 'importance' for either:
    - LinearRegression: absolute standardized coefficients
    - Tree/Ensemble: feature_importances_
    """
    feat_names = get_feature_names(pipeline)
    est = pipeline.named_steps["regressor"]

    if hasattr(est, "coef_"):  # LinearRegression
        coefs = np.ravel(est.coef_)
        imp = np.abs(coefs)  # use absolute value for "importance-like" view
    elif hasattr(est, "feature_importances_"):  # RF/GB
        imp = est.feature_importances_
    else:
        raise ValueError("Model does not expose coefficients or feature_importances_")

    df_imp = pd.DataFrame({"feature": feat_names, "importance": imp})
    df_imp.sort_values("importance", ascending=False, inplace=True)
    return df_imp


def plot_feature_importance(pipeline, title, top_n=20, out_path="figures/feature_importance.png"):
    _ensure_figs()
    df_imp = feature_importance_df(pipeline).head(top_n)

    plt.figure(figsize=(8, max(4, 0.35*len(df_imp))))
    sns.barplot(y="feature", x="importance", data=df_imp)
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_actual_vs_pred(y_true, y_pred, title, out_path="figures/actual_vs_pred.png"):
    _ensure_figs()
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_residuals(y_true, y_pred, prefix="figures/residuals"):
    _ensure_figs()
    resid = y_true - y_pred

    # Residuals vs Predicted
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=y_pred, y=resid, alpha=0.6)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted Charges")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(f"{prefix}_scatter.png")
    plt.close()

    # Residual distribution
    plt.figure(figsize=(7,5))
    sns.histplot(resid, bins=30, kde=True)
    plt.xlabel("Residual")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(f"{prefix}_hist.png")
    plt.close()


def plot_group_differences(df):
    """
    Group visuals: charges by smoker, region, and smoker x region.
    """
    _ensure_figs()

    # Mean charges by smoker
    plt.figure(figsize=(6,4))
    sns.barplot(x="smoker", y="charges", data=df, estimator=np.mean, errorbar="se")
    plt.title("Average Charges by Smoking Status")
    plt.tight_layout()
    plt.savefig("figures/avg_charges_by_smoker.png")
    plt.close()

    # Mean charges by region
    plt.figure(figsize=(7,4))
    sns.barplot(x="region", y="charges", data=df, estimator=np.mean, errorbar="se")
    plt.title("Average Charges by Region")
    plt.tight_layout()
    plt.savefig("figures/avg_charges_by_region.png")
    plt.close()

    # Boxplot smoker vs charges (distribution)
    plt.figure(figsize=(6,4))
    sns.boxplot(x="smoker", y="charges", data=df)
    plt.title("Charges Distribution by Smoking Status")
    plt.tight_layout()
    plt.savefig("figures/box_charges_by_smoker.png")
    plt.close()

    # Optional: interaction smoker x region
    plt.figure(figsize=(7.5,4.5))
    sns.barplot(x="region", y="charges", hue="smoker", data=df, estimator=np.mean, errorbar="se")
    plt.title("Average Charges by Region & Smoking Status")
    plt.tight_layout()
    plt.savefig("figures/avg_charges_region_smoker.png")
    plt.close()
