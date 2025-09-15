import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

def ensure_folders():
    Path("data").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

def download_dataset():
    load_dotenv()  # loads KAGGLE_USERNAME/KAGGLE_KEY from .env
    print("Kaggle username:", os.getenv("KAGGLE_USERNAME"))

    api = KaggleApi()
    api.authenticate()

    ensure_folders()
    api.dataset_download_files(
        "mosapabdelghany/medical-insurance-cost-dataset",
        path="data",
        unzip=True
    )
    print("Dataset downloaded to data/")

if __name__ == "__main__":
    download_dataset()
