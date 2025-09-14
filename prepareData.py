# preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor():
    categorical = ["sex", "smoker", "region"]
    numeric = ["age", "bmi", "children"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    return preprocessor
