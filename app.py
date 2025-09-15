import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------
# Cache: data & modeller
# ----------------------------
@st.cache_data
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # säkerställ förväntade kolumner
    expected = {"age","sex","bmi","children","smoker","region","charges"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Saknade kolumner i CSV: {missing}")
    return df

def build_preprocessor():
    categorical = ["sex", "smoker", "region"]
    numeric = ["age", "bmi", "children"]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    return pre, categorical + numeric

@st.cache_resource
def train_pipeline(df: pd.DataFrame, model_name: str, params: dict):
    X = df.drop(columns=["charges"])
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params.get("test_size", 0.2), random_state=42
    )

    pre, _ = build_preprocessor()

    if model_name == "Linear Regression":
        reg = LinearRegression()
    elif model_name == "Random Forest":
        reg = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", None),
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "Gradient Boosting":
        reg = GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=42,
        )
    else:
        raise ValueError("Okänd modell")

    pipe = Pipeline(steps=[("preprocess", pre), ("regressor", reg)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R²": float(r2_score(y_test, y_pred)),
    }
    return pipe, X_test, y_test, y_pred, metrics


# ----------------------------
# Hjälpfunktioner för plots
# ----------------------------
def transformed_feature_names(pipeline: Pipeline):
    pre = pipeline.named_steps["preprocess"]
    return pre.get_feature_names_out()

def feature_importance_df(pipeline: Pipeline) -> pd.DataFrame:
    names = transformed_feature_names(pipeline)
    reg = pipeline.named_steps["regressor"]
    if hasattr(reg, "coef_"):
        vals = np.abs(np.ravel(reg.coef_))  # absolutvärde för "importance"
    elif hasattr(reg, "feature_importances_"):
        vals = reg.feature_importances_
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    imp = pd.DataFrame({"feature": names, "importance": vals})
    return imp.sort_values("importance", ascending=False)

def plot_actual_vs_pred_interactive(y_true, y_pred, title):
    dfp = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    fig = px.scatter(dfp, x="Actual", y="Predicted", title=title)
    lo, hi = dfp.min().min(), dfp.max().max()
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(color="red", dash="dash"))
    return fig

def plot_residuals_interactive(y_true, y_pred):
    resid = y_true - y_pred
    fig1 = px.scatter(x=y_pred, y=resid, labels={"x":"Predicted", "y":"Residual"},
                      title="Residuals vs Predicted")
    fig1.add_hline(y=0, line_dash="dash", line_color="red")
    fig2 = px.histogram(resid, nbins=30, title="Residual Distribution")
    return fig1, fig2


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Insurance Charges Studio", page_icon="💊", layout="wide")
st.title("💊 Medical Insurance — Explorera & Prediktera kostnader")

# Data-laddning
csv_path = Path("data/insurance.csv")
if not csv_path.exists():
    st.error("Hittar inte `data/insurance.csv`. Lägg filen i `data/` eller ladda ner från Kaggle.")
    st.stop()

df = load_data(csv_path)

# Sidebar: navigering
page = st.sidebar.radio("Navigera", ["Utforska data", "Träna modell", "Prediktion"])

# ----------------------------
# SIDA 1: Utforska data
# ----------------------------
if page == "Utforska data":
    st.subheader("📦 Dataset")
    with st.expander("Visa första raderna"):
        st.dataframe(df.head(20), use_container_width=True)

    # Filter
    st.subheader("🔎 Filter")
    c1, c2, c3 = st.columns(3)
    with c1:
        smoker_filter = st.multiselect("Smoker", options=sorted(df["smoker"].unique()),
                                       default=list(df["smoker"].unique()))
    with c2:
        region_filter = st.multiselect("Region", options=sorted(df["region"].unique()),
                                       default=list(df["region"].unique()))
    with c3:
        sex_filter = st.multiselect("Sex", options=sorted(df["sex"].unique()),
                                    default=list(df["sex"].unique()))

    age_min, age_max = int(df["age"].min()), int(df["age"].max())
    bmi_min, bmi_max = float(df["bmi"].min()), float(df["bmi"].max())
    age_range = st.slider("Ålder", min_value=age_min, max_value=age_max,
                          value=(age_min, age_max))
    bmi_range = st.slider("BMI", min_value=float(np.floor(bmi_min)),
                          max_value=float(np.ceil(bmi_max)),
                          value=(float(np.floor(bmi_min)), float(np.ceil(bmi_max))))

    dff = df[
        (df["smoker"].isin(smoker_filter)) &
        (df["region"].isin(region_filter)) &
        (df["sex"].isin(sex_filter)) &
        (df["age"].between(age_range[0], age_range[1])) &
        (df["bmi"].between(bmi_range[0], bmi_range[1]))
    ]

    st.caption(f"Rader efter filter: **{len(dff)}**")

    tab1, tab2, tab3 = st.tabs(["Fördelningar", "Gruppjämförelser", "Korrelationsvy"])
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(px.histogram(dff, x="age", nbins=20, title="Age"), use_container_width=True)
        with c2:
            st.plotly_chart(px.histogram(dff, x="bmi", nbins=20, title="BMI"), use_container_width=True)
        with c3:
            st.plotly_chart(px.histogram(dff, x="charges", nbins=30, title="Charges"), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.box(dff, x="smoker", y="charges", title="Charges by Smoking Status"),
                            use_container_width=True)
        with c2:
            st.plotly_chart(px.box(dff, x="region", y="charges", title="Charges by Region"),
                            use_container_width=True)
        st.plotly_chart(px.bar(dff, x="region", y="charges", color="smoker",
                               barmode="group",
                               title="Average Charges by Region & Smoker",
                               category_orders={"region":sorted(df["region"].unique())},
                               labels={"charges":"Mean charges"}).update_traces(opacity=0.9)
                          .update_layout(xaxis_title="region").update_traces(error_y=None),
                          use_container_width=True)

    with tab3:
        num_cols = ["age","bmi","children","charges"]
        corr = dff[num_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", origin="lower",
                        title="Korrelationsmatris (numeriska)")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# SIDA 2: Träna modell
# ----------------------------
elif page == "Träna modell":
    st.subheader("⚙️ Träning & utvärdering")

    model_name = st.selectbox("Välj modell", ["Linear Regression", "Random Forest", "Gradient Boosting"])

    params = {"test_size": st.slider("Teststorlek", 0.1, 0.4, 0.2, 0.05)}
    if model_name in ("Random Forest", "Gradient Boosting"):
        params["n_estimators"] = st.slider("n_estimators", 50, 500, 300, 50)
    if model_name == "Random Forest":
        params["max_depth"] = st.slider("max_depth (None = obegränsad)", 1, 20, 0, 1)
        if params["max_depth"] == 0:
            params["max_depth"] = None
    if model_name == "Gradient Boosting":
        params["learning_rate"] = st.slider("learning_rate", 0.01, 0.5, 0.1, 0.01)
        params["max_depth"] = st.slider("max_depth (trädens djup)", 1, 6, 3, 1)

    if st.button("Träna modell", type="primary"):
        pipe, X_test, y_test, y_pred, metrics = train_pipeline(df, model_name, params)
        st.session_state["last_model"] = (model_name, pipe, X_test, y_test, y_pred, metrics)

    if "last_model" in st.session_state:
        name, pipe, X_test, y_test, y_pred, metrics = st.session_state["last_model"]
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{metrics['MAE']:.0f}")
        c2.metric("RMSE", f"{metrics['RMSE']:.0f}")
        c3.metric("R²", f"{metrics['R²']:.3f}")

        st.markdown("### 📈 Actual vs Predicted")
        st.plotly_chart(plot_actual_vs_pred_interactive(y_test, y_pred, f"Actual vs Predicted — {name}"),
                        use_container_width=True)

        st.markdown("### 🔎 Residualer")
        r1, r2 = plot_residuals_interactive(y_test, y_pred)
        st.plotly_chart(r1, use_container_width=True)
        st.plotly_chart(r2, use_container_width=True)

        st.markdown("### 🧠 Feature importance")
        imp = feature_importance_df(pipe)
        if imp.empty:
            st.info("Modellen exponerar inte feature importance.")
        else:
            fig_imp = px.bar(imp.head(20), x="importance", y="feature", orientation="h",
                             title=f"Feature importance — {name}")
            st.plotly_chart(fig_imp, use_container_width=True)

# ----------------------------
# SIDA 3: Prediktion
# ----------------------------
elif page == "Prediktion":
    st.subheader("🧮 Prediktera försäkringskostnad")

    # Snabb modell i bakgrunden (Linear Regression default) om ingen tränad ännu
    if "last_model" not in st.session_state:
        pipe, *_ = train_pipeline(df, "Linear Regression", {"test_size": 0.2})
        st.session_state["last_model"] = ("Linear Regression", pipe, None, None, None, {})

    name, pipe, *_ = st.session_state["last_model"]

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)
    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=26.5, step=0.1)
        sex = st.selectbox("Sex", options=sorted(df["sex"].unique()), index=0)
    with col3:
        smoker = st.selectbox("Smoker", options=sorted(df["smoker"].unique()))
        region = st.selectbox("Region", options=sorted(df["region"].unique()))

    input_df = pd.DataFrame([{
        "age": age, "bmi": bmi, "children": children,
        "sex": sex, "smoker": smoker, "region": region
    }])

    pred = float(pipe.predict(input_df)[0])
    st.metric("Predikterad kostnad (charges)", f"{pred:,.0f}")

    with st.expander("Visa indata (modellens features)"):
        st.write(input_df)
