# train_models.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prepareData import build_preprocessor  # your existing preprocessor


def _metrics(y_true, y_pred):
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def train_models(df):
    """
    Trains 3 models (Linear, RandomForest, GradientBoosting) on the insurance dataset.
    Returns a dict of fitted pipelines + predictions + metrics.
    """
    X = df.drop(columns=["charges"])
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, random_state=42),
    }

    results = {}
    for name, reg in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("regressor", reg)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[name] = {
            "pipeline": pipe,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "metrics": _metrics(y_test, y_pred),
        }

    return results
