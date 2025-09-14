from pathlib import Path
import pandas as pd

from eda import quick_overview, check_missing_and_corr, dist_plots
from prepareData import build_preprocessor 
from train import train_and_evaluate

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
    train_and_evaluate(df)
