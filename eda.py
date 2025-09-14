# eda.py
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def _ensure_figures_dir():
    Path("figures").mkdir(exist_ok=True)

def quick_overview(df):
    print("\n--- First rows ---")
    print(df.head())

    print("\n--- Shape ---")
    print(df.shape)

    print("\n--- Info ---")
    print(df.info())

    print("\n--- Summary statistics ---")
    print(df.describe(include="all"))

    print("\n--- Unique values (categorical) ---")
    for col in df.columns:
        if df[col].dtype == "object":
            print(f"{col}: {df[col].unique()}")

def check_missing_and_corr(df):
    print("\n--- Missing values per column ---")
    print(df.isnull().sum())

    print("\n--- Percentage missing values ---")
    print(df.isnull().mean() * 100)

    print("\n--- Correlations (numeric) ---")
    print(df.corr(numeric_only=True))

def dist_plots(df):
    _ensure_figures_dir()

    # Histograms
    plt.figure(figsize=(6,4))
    sns.histplot(df["age"], bins=20, kde=True)
    plt.title("Distribution of Age")
    plt.xlabel("Age"); plt.ylabel("Count")
    plt.savefig("figures/hist_age.png"); plt.close()

    plt.figure(figsize=(6,4))
    sns.histplot(df["bmi"], bins=20, kde=True, color="orange")
    plt.title("Distribution of BMI")
    plt.xlabel("BMI"); plt.ylabel("Count")
    plt.savefig("figures/hist_bmi.png"); plt.close()

    plt.figure(figsize=(6,4))
    sns.histplot(df["charges"], bins=30, kde=True, color="green")
    plt.title("Distribution of Insurance Charges")
    plt.xlabel("Charges"); plt.ylabel("Count")
    plt.savefig("figures/hist_charges.png"); plt.close()

    # Boxplots
    plt.figure(figsize=(6,2))
    sns.boxplot(x=df["age"])
    plt.title("Boxplot of Age")
    plt.savefig("figures/box_age.png"); plt.close()

    plt.figure(figsize=(6,2))
    sns.boxplot(x=df["bmi"], color="orange")
    plt.title("Boxplot of BMI")
    plt.savefig("figures/box_bmi.png"); plt.close()

    plt.figure(figsize=(6,2))
    sns.boxplot(x=df["charges"], color="green")
    plt.title("Boxplot of Insurance Charges")
    plt.savefig("figures/box_charges.png"); plt.close()

    # Group comparison
    plt.figure(figsize=(6,4))
    sns.boxplot(x="smoker", y="charges", data=df)
    plt.title("Charges by Smoking Status")
    plt.savefig("figures/box_charges_by_smoker.png"); plt.close()
