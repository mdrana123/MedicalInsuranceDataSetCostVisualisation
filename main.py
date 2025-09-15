from pathlib import Path
import pandas as pd

from eda import quick_overview, check_missing_and_corr, dist_plots
from prepareData import build_preprocessor 
from train_models import train_models
from visualize import (
    plot_feature_importance, plot_actual_vs_pred, plot_residuals, plot_group_differences
)

def load_data():
    csv_path = Path("data/insurance.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            "data/insurance.csv not found. Run `python download_data.py` first, "
            "or place the CSV in the data/ folder."
        )
    return pd.read_csv(csv_path)

if __name__ == "__main__":
    df = load_data()
    quick_overview(df)
    check_missing_and_corr(df)
    dist_plots(df)
    print("Saved plots in the figures/ folder.")
    preprocessor= build_preprocessor()
    print("Preprocessor built successfully.")
    results = train_models(df)
    # For each model, save: feature importance, actual vs predicted, residuals
    for name, res in results.items():
        pipe = res["pipeline"]
        y_test = res["y_test"]
        y_pred = res["y_pred"]

        print(f"\n{name} metrics:", res["metrics"])

        # Feature importance
        plot_feature_importance(
            pipe,
            title=f"Feature Importance — {name}",
            out_path=f"figures/feature_importance_{name}.png"
        )

        # Actual vs Predicted
        plot_actual_vs_pred(
            y_test, y_pred,
            title=f"Actual vs Predicted — {name}",
            out_path=f"figures/actual_vs_pred_{name}.png"
        )

        # Residuals
        plot_residuals(
            y_test, y_pred,
            prefix=f"figures/residuals_{name}"
        )

    print("\nSaved all figures in the figures/ folder.")
    
