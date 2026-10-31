from __future__ import annotations

from pathlib import Path

import joblib
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DATA_PATH = DATA_DIR / "city_day.csv"
CITY_DATA_DIR = ROOT / "airapi_model"
MODEL_PATHS = {
    "XGBoost": ROOT / "Saved_models" / "xgb_aqi_model.joblib",
    "Random Forest": ROOT / "Saved_models" / "rf_aqi_model.joblib",
    "LightGBM": ROOT / "Saved_models" / "lgbm_aqi_model.joblib",
}

FEATURES = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene",
]

CLASS_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def get_kaggle_data_path() -> Path:
    if not LOCAL_DATA_PATH.exists():
        dataset_path = Path(kagglehub.dataset_download("rohanrao/air-quality-data-in-india"))
        source_path = dataset_path / "city_day.csv"
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset file not found in Kaggle download: {dataset_path}")
        LOCAL_DATA_PATH.write_bytes(source_path.read_bytes())
    return LOCAL_DATA_PATH


@st.cache_resource
def load_model(model_name: str):
    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


@st.cache_data
def load_raw_dataset():
    df = pd.read_csv(get_kaggle_data_path())
    df = df.drop(columns=["City", "Date"], errors="ignore")
    df = df.dropna(subset=["AQI"]).copy()
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_dataset():
    df = load_raw_dataset()
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    return df


@st.cache_data
def get_missing_value_summary():
    df = load_raw_dataset()
    pollutant_cols = [col for col in FEATURES if col in df.columns]
    missing = df[pollutant_cols].isna().sum().sort_values(ascending=False).reset_index()
    missing.columns = ["Pollutant", "Missing Values"]
    return missing


@st.cache_data
def load_city_monthly_aqi():
    if not CITY_DATA_DIR.exists():
        return pd.DataFrame(columns=["city", "month", "AQI"])

    frames = []
    for csv_file in sorted(CITY_DATA_DIR.glob("*.csv")):
        try:
            city_df = pd.read_csv(csv_file)
            if "local" not in city_df.columns or "parameter" not in city_df.columns or "value" not in city_df.columns:
                continue
            city_df = city_df.copy()
            city_df["local"] = pd.to_datetime(city_df["local"], errors="coerce")
            city_df = city_df.dropna(subset=["local", "parameter", "value"])
            city_df["parameter"] = city_df["parameter"].str.lower()
            city_df["value"] = pd.to_numeric(city_df["value"], errors="coerce")
            if city_df.empty:
                continue

            pivot = city_df.pivot_table(index="local", columns="parameter", values="value", aggfunc="mean")
            pm25 = pd.to_numeric(pivot.get("pm25", 0), errors="coerce").fillna(0)
            pm10 = pd.to_numeric(pivot.get("pm10", 0), errors="coerce").fillna(0)
            local_df = pd.DataFrame({"local": pivot.index, "PM25": pm25, "PM10": pm10})
            local_df["AQI"] = (local_df["PM25"] * 2.5 + local_df["PM10"] * 1.3).clip(lower=0)
            city_name = city_df["city"].dropna().iloc[0]
            local_df["city"] = city_name
            local_df["month"] = local_df["local"].dt.to_period("M").astype(str)
            summary = local_df.groupby(["city", "month"], as_index=False)["AQI"].mean()
            frames.append(summary)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["city", "month", "AQI"])
    return pd.concat(frames, ignore_index=True)


def classify_aqi(aqi_value: float) -> str:
    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Satisfactory"
    if aqi_value <= 200:
        return "Moderate"
    if aqi_value <= 300:
        return "Poor"
    if aqi_value <= 400:
        return "Very Poor"
    return "Severe"


@st.cache_data
def prepare_regression_data(df: pd.DataFrame):
    feature_cols = [column for column in FEATURES if column in df.columns]
    X = df[feature_cols]
    y = df["AQI"]
    return feature_cols, train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


@st.cache_data
def load_city_options():
    df = pd.read_csv(get_kaggle_data_path())
    cities = df["City"].dropna().astype(str).unique().tolist()
    return sorted(cities)


@st.cache_data
def load_city_regression_data(city_name: str):
    df = pd.read_csv(get_kaggle_data_path())
    city_df = df[df["City"].astype(str) == city_name].copy()
    if city_df.empty:
        raise ValueError(f"No data available for city: {city_name}")

    if "Date" in city_df.columns:
        city_df["Date"] = pd.to_datetime(city_df["Date"], errors="coerce")
    city_df = city_df.sort_values("Date").dropna(subset=["AQI"]).copy()
    for col in FEATURES:
        if col in city_df.columns:
            city_df[col] = pd.to_numeric(city_df[col], errors="coerce")
    city_df[FEATURES] = city_df[FEATURES].fillna(city_df[FEATURES].median())
    return city_df.reset_index(drop=True)


@st.cache_data
def evaluate_model_performance(model_name: str, city_name: str | None = None):
    if city_name is not None:
        df = load_city_regression_data(city_name)
    else:
        df = load_dataset()

    feature_cols, (X_train, X_test, y_train, y_test) = prepare_regression_data(df)
    model = load_model(model_name)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if city_name is not None:
        dates = df.loc[X_test.index, "Date"]
        actual = pd.Series(y_test.to_numpy(), index=dates)
        forecast = pd.Series(y_pred, index=dates)
        actual = actual.sort_index()
        forecast = forecast.sort_index()
    else:
        actual = y_test
        forecast = pd.Series(y_pred, index=y_test.index)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "y_test": actual,
        "y_pred": forecast,
        "feature_cols": feature_cols,
    }


@st.cache_data
def benchmark_regression_models():
    df = load_dataset()
    feature_cols, (X_train, X_test, y_train, y_test) = prepare_regression_data(df)

    models = {
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            objective="regression",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(
            {
                "Model": name,
                "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "MAE": float(mean_absolute_error(y_test, y_pred)),
                "R2": float(r2_score(y_test, y_pred)),
            }
        )

    df_results = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    best_model_name = df_results.iloc[0]["Model"]
    return df_results, best_model_name


@st.cache_data
def benchmark_classification_models():
    df = load_dataset().copy()
    df["AQI_Bucket"] = df["AQI"].apply(classify_aqi)
    X = df[FEATURES]
    y = df["AQI_Bucket"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(
            {
                "Model": name,
                "Weighted F1": float(f1_score(y_test, y_pred, average="weighted")),
                "Accuracy": float(accuracy_score(y_test, y_pred)),
            }
        )

    df_results = pd.DataFrame(results).sort_values("Weighted F1", ascending=False).reset_index(drop=True)
    best_model_name = df_results.iloc[0]["Model"]
    return df_results, best_model_name


@st.cache_data
def get_best_regression_importance(best_model_name: str):
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df["AQI"], test_size=0.2, random_state=42)

    if best_model_name == "XGBoost":
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
    elif best_model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    model.fit(X_train, y_train)
    importances = model.feature_importances_
    return pd.Series(importances, index=FEATURES).sort_values(ascending=False)


@st.cache_data
def get_classification_summary():
    df = load_dataset().copy()
    df["AQI_Bucket"] = df["AQI"].apply(classify_aqi)
    X = df[FEATURES]
    y = df["AQI_Bucket"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return cm, CLASS_ORDER, f1


@st.cache_data
def get_city_cluster_summary():
    df = load_dataset()
    cluster_features = [col for col in FEATURES if col in df.columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cluster_features])

    k_values = range(2, 9)
    inertia = []
    silhouette_scores = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertia.append((k, model.inertia_))
        silhouette_scores.append((k, silhouette_score(X_scaled, labels)))

    best_k = max(silhouette_scores, key=lambda item: item[1])[0]
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
    pca_df["Cluster"] = labels.astype(str)

    return {
        "inertia": inertia,
        "silhouette": silhouette_scores,
        "best_k": best_k,
        "pca_df": pca_df,
    }


def plot_missing_values(df: pd.DataFrame):
    pollutant_cols = [col for col in FEATURES if col in df.columns]
    missing = df[pollutant_cols].isna().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0].reset_index()
    missing.columns = ["Pollutant", "Missing %"]

    if missing.empty:
        st.info("No missing values were found in the dataset.")
        return

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    ax.bar(missing["Pollutant"], missing["Missing %"], color="#4c78a8")
    ax.set_title("Percentage of Missing Values per Pollutant Before Imputation")
    ax.set_ylabel("Missing (%)")
    ax.set_xlabel("Pollutant")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(pad=1)
    st.pyplot(fig)


def plot_health_distribution(df: pd.DataFrame):
    df_copy = df.copy()
    df_copy["Health Category"] = df_copy["AQI"].apply(classify_aqi)
    counts = df_copy["Health Category"].value_counts().reindex(CLASS_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(5.3, 3.1))
    counts.plot(kind="bar", ax=ax, color=["#A6D96A", "#FEE08B", "#FDAE61", "#F46D43", "#D73027", "#7F0000"])
    ax.set_title("Distribution of AQI Health Categories")
    ax.set_xlabel("Health Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(pad=1)
    st.pyplot(fig)


def plot_correlation_heatmap(df: pd.DataFrame):
    corr = df[[*FEATURES, "AQI"]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    img = ax.imshow(corr, cmap="coolwarm")
    ax.set_title("Correlation Heatmap of Pollutants and AQI")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=1)
    st.pyplot(fig)


def plot_monthly_city_aqi():
    city_summary = load_city_monthly_aqi()
    if city_summary.empty:
        st.warning("City AQI trend data is not available in the airapi_model folder.")
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for city_name, group in city_summary.groupby("city"):
        city_series = group.sort_values("month")
        ax.plot(city_series["month"], city_series["AQI"], marker="o", linewidth=2, label=city_name)
    ax.set_title("Average Monthly AQI of the Cities")
    ax.set_xlabel("Month")
    ax.set_ylabel("AQI")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: pd.Series, model_name: str):
    if isinstance(y_true.index, pd.DatetimeIndex):
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        ax.plot(y_true.index, y_true.values, label="Actual AQI", linewidth=2)
        ax.plot(y_pred.index, y_pred.values, label="Forecast AQI", linewidth=2, linestyle="--")
        ax.set_title(f"Actual vs Forecast AQI for {model_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("AQI")
        ax.legend()
        plt.xticks(rotation=45)
    else:
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.scatter(y_true, y_pred, alpha=0.7)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", lw=1.5)
        ax.set_title(f"Actual vs Predicted AQI ({model_name})")
        ax.set_xlabel("Actual AQI")
        ax.set_ylabel("Predicted AQI")
    plt.tight_layout()
    st.pyplot(fig)


def plot_feature_importance(model_name: str):
    importance = get_best_regression_importance(model_name)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    importance.head(10).sort_values().plot(kind="barh", color="#2E86AB", ax=ax)
    ax.set_title(f"Feature Importance of the Best Regression Model ({model_name})")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout(pad=1)
    st.pyplot(fig)


def plot_confusion_matrix():
    cm, labels, f1 = get_classification_summary()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    img = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix of the Best Classifier")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=1)
    st.pyplot(fig)
    st.caption(f"Weighted F1-score: {f1:.4f}")


def plot_elbow_curve():
    cluster_summary = get_city_cluster_summary()
    elbow_points = pd.DataFrame(cluster_summary["inertia"], columns=["k", "inertia"])
    silhouette_points = pd.DataFrame(cluster_summary["silhouette"], columns=["k", "silhouette"])

    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.1))
    ax[0].plot(elbow_points["k"], elbow_points["inertia"], marker="o")
    ax[0].set_title("Elbow Curve for Choosing k")
    ax[0].set_xlabel("Number of clusters (k)")
    ax[0].set_ylabel("Inertia")

    ax[1].plot(silhouette_points["k"], silhouette_points["silhouette"], marker="s", color="green")
    ax[1].set_title("Silhouette Scores by k")
    ax[1].set_xlabel("Number of clusters (k)")
    ax[1].set_ylabel("Silhouette Score")
    plt.tight_layout()
    st.pyplot(fig)


def plot_cluster_pca():
    cluster_summary = get_city_cluster_summary()
    pca_df = cluster_summary["pca_df"]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for cluster_id in sorted(pca_df["Cluster"].unique(), key=lambda x: int(x)):
        subset = pca_df[pca_df["Cluster"] == cluster_id]
        ax.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster_id}")
    ax.set_title("PCA Scatter Plot of the City Clusters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)


st.set_page_config(page_title="Air Quality Prediction", layout="centered")
st.title("Air Quality Forecasting & AQI Prediction")
st.caption("Multi-modal air-quality monitoring and forecasting dashboard")

with st.sidebar:
    st.header("Model Controls")
    model_name = st.selectbox("Select regression model", list(MODEL_PATHS.keys()), index=0)
    st.markdown("---")
    st.header("Live AQI Input")
    values = {
        "PM2.5": st.number_input("PM2.5 (µg/m³)", value=40.0, min_value=0.0, max_value=500.0, step=1.0),
        "PM10": st.number_input("PM10 (µg/m³)", value=70.0, min_value=0.0, max_value=500.0, step=1.0),
        "NO": st.number_input("NO", value=20.0, min_value=0.0, max_value=500.0, step=1.0),
        "NO2": st.number_input("NO2", value=25.0, min_value=0.0, max_value=500.0, step=1.0),
        "NOx": st.number_input("NOx", value=45.0, min_value=0.0, max_value=500.0, step=1.0),
        "NH3": st.number_input("NH3", value=15.0, min_value=0.0, max_value=250.0, step=1.0),
        "CO": st.number_input("CO", value=0.8, min_value=0.0, max_value=20.0, step=0.1),
        "SO2": st.number_input("SO2", value=15.0, min_value=0.0, max_value=200.0, step=1.0),
        "O3": st.number_input("O3", value=35.0, min_value=0.0, max_value=200.0, step=1.0),
        "Benzene": st.number_input("Benzene", value=3.0, min_value=0.0, max_value=50.0, step=0.1),
        "Toluene": st.number_input("Toluene", value=6.0, min_value=0.0, max_value=50.0, step=0.1),
        "Xylene": st.number_input("Xylene", value=2.0, min_value=0.0, max_value=50.0, step=0.1),
    }
    submitted = st.button("Predict AQI")


def show_overview_tab():
    raw_df = load_raw_dataset()
    df = load_dataset()
    st.subheader("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Mean AQI", round(float(df["AQI"].mean()), 2))
    with col3:
        st.metric("Max AQI", int(df["AQI"].max()))
    with col4:
        st.metric("Min AQI", int(df["AQI"].min()))

    st.dataframe(df.describe().T, width="stretch")

    st.markdown("### Missing values before imputation")
    st.dataframe(get_missing_value_summary(), width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        plot_missing_values(raw_df)
    with col_b:
        plot_health_distribution(df)

    st.markdown("---")
    plot_correlation_heatmap(df)

    st.markdown("---")
    plot_monthly_city_aqi()


def show_forecasting_tab():
    city_options = load_city_options()
    selected_city = st.selectbox("Select a city", city_options, index=0)
    reg_metrics = evaluate_model_performance(model_name, selected_city)
    benchmark_df, best_model = benchmark_regression_models()
    st.subheader(f"Regression Model Benchmark for {selected_city}")
    st.dataframe(benchmark_df, width="stretch")
    st.caption(f"Best regression model: {best_model}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RMSE", round(float(reg_metrics["rmse"]), 4))
    with col2:
        st.metric("MAE", round(float(reg_metrics["mae"]), 4))
    with col3:
        st.metric("R²", round(float(reg_metrics["r2"]), 4))

    st.markdown("---")
    plot_actual_vs_predicted(reg_metrics["y_test"], reg_metrics["y_pred"], model_name)

    st.markdown("---")
    plot_feature_importance(best_model)


def show_classification_tab():
    benchmark_df, best_model = benchmark_classification_models()
    st.subheader("Classification Model Benchmark")
    st.dataframe(benchmark_df, width="stretch")
    st.caption(f"Best classifier: {best_model}")

    st.markdown("---")
    plot_confusion_matrix()


def show_clusters_tab():
    plot_elbow_curve()
    st.markdown("---")
    plot_cluster_pca()


def show_live_prediction_tab():
    form_values = {key: values[key] for key in FEATURES}

    if submitted:
        model = load_model(model_name)
        feature_row = pd.DataFrame([form_values], columns=FEATURES)
        prediction = model.predict(feature_row)[0]
        risk = classify_aqi(prediction)

        st.subheader("Prediction Result")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted AQI", round(float(prediction), 2))
        with col2:
            st.metric("Health Risk", risk)
        with col3:
            st.metric("Action", "Monitor" if prediction < 100 else "Action Needed")

        if prediction <= 50:
            advisory = "Air quality is good. Suitable for outdoor activity."
        elif prediction <= 100:
            advisory = "Air quality is satisfactory. Sensitive groups may still be affected."
        elif prediction <= 200:
            advisory = "Air quality is moderate. Reduce prolonged outdoor exposure if sensitive."
        elif prediction <= 300:
            advisory = "Air quality is poor. Limit time outdoors and use protective masks when necessary."
        elif prediction <= 400:
            advisory = "Air quality is very poor. Avoid prolonged outdoor exposure."
        else:
            advisory = "Air quality is severe. Stay indoors and avoid exertion."

        st.info(advisory)
    else:
        st.info("Choose values from the sidebar and press Predict AQI.")


overview_tab, forecasting_tab, classification_tab, clusters_tab, prediction_tab = st.tabs(
    ["Overview", "AQI Forecasting", "Health Risk Classification", "City Clusters", "Live Prediction"]
)

with overview_tab:
    show_overview_tab()

with forecasting_tab:
    show_forecasting_tab()

with classification_tab:
    show_classification_tab()

with clusters_tab:
    show_clusters_tab()

with prediction_tab:
    show_live_prediction_tab()
