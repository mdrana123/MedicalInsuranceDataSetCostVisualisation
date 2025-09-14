import os
from dotenv import load_dotenv

load_dotenv()

print("KAGGLE_USERNAME =", os.getenv("KAGGLE_USERNAME"))
print("KAGGLE_KEY =", os.getenv("KAGGLE_KEY")[:5] + "*****")
