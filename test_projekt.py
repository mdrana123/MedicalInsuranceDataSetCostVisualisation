import numpy as np
import pandas as pd
import pytest

from prepareData import build_preprocessor


def _train_df():
    return pd.DataFrame({
        "age":      [23, 45, 34, 50, 29, 41],
        "bmi":      [28.5, 31.2, 22.0, 27.1, 35.3, 30.4],
        "children": [0,   2,    1,    3,    0,    1],
        "smoker":   ["no","yes","no","no","yes","no"],
        "sex":      ["female","male","female","male","female","male"],
        "region":   ["southwest","southeast","northwest","northeast","southeast","southwest"],
        "charges":  [3200.0, 45000.0, 9000.0, 16000.0, 28000.0, 12000.0],
    })


def test_preprocessor_structure():
    ct = build_preprocessor()

    # names are available without fitting
    names = [name for name, _, _ in ct.transformers]
    assert "cat" in names and "num" in names

    # extract actual transformers from the 3-tuples
    ohe = next(trans for name, trans, _ in ct.transformers if name == "cat")
    scaler = next(trans for name, trans, _ in ct.transformers if name == "num")

    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    assert isinstance(ohe, OneHotEncoder)
    assert ohe.drop == "first"
    assert isinstance(scaler, StandardScaler)


def test_fit_transform_shape_and_dtype():
    df = _train_df()
    X = df.drop(columns=["charges"])

    ct = build_preprocessor()
    Xt = ct.fit_transform(X)  # after this, named_transformers_ exists

    # compute expected number of OHE columns from fitted encoder
    ohe = ct.named_transformers_["cat"]
    num_ohe_cols = sum(len(cats) - 1 for cats in ohe.categories_)
    n_numeric = 3  # age, bmi, children
    total_expected = num_ohe_cols + n_numeric

    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] == total_expected
    assert np.issubdtype(Xt.dtype, np.number)


def test_unseen_category_raises_value_error():
    """
    With OneHotEncoder(drop='first') and no handle_unknown='ignore',
    unseen categories should raise ValueError on transform.
    """
    train = _train_df().drop(columns=["charges"])
    test = pd.DataFrame({
        "age": [37, 52],
        "bmi": [29.0, 33.1],
        "children": [1, 2],
        "smoker": ["no", "yes"],
        "sex": ["male", "female"],
        "region": ["midwest", "midwest"],  # unseen
    })

    ct = build_preprocessor()
    ct.fit(train)

    with pytest.raises(ValueError):
        ct.transform(test)
