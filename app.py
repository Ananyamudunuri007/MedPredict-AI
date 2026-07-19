import numpy as np
import pandas as pd
import plotly.express as px
import joblib
import streamlit as st
from pathlib import Path
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# App configuration and styling
# ------------------------------------------------------------
st.set_page_config(
    page_title="MedPredict AI",
    page_icon="🏥",
    layout="wide",
)

# Apply a polished visual theme to the app.
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e3a8a 100%);
        color: white;
    }
    .block-container {
        padding-top: 1rem;
    }
    .hero-card, .info-card {
        background: white;
        padding: 1.15rem 1.2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .hero-card h3, .info-card h3 {
        margin-top: 0;
        color: #0f172a;
    }
    .stButton > button {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Data and model loading helpers
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


@st.cache_resource
def load_saved_models():
    """Load the trained machine learning models from the models folder."""
    cancer_model = joblib.load(MODEL_DIR / "breast_cancer.pkl")
    diabetes_model = joblib.load(MODEL_DIR / "diabetes.pkl")
    return cancer_model, diabetes_model


@st.cache_data
def load_datasets():
    """Load the sklearn datasets used for the UI and visualizations."""
    cancer_dataset = load_breast_cancer()
    diabetes_dataset = load_diabetes()
    return cancer_dataset, diabetes_dataset


@st.cache_data
def get_model_metrics():
    """Compute lightweight comparison metrics to show on the model comparison page."""
    cancer_dataset, diabetes_dataset = load_datasets()
    cancer_model, diabetes_model = load_saved_models()

    # Breast cancer accuracy
    cancer_train, cancer_test, y_train, y_test = train_test_split(
        cancer_dataset.data,
        cancer_dataset.target,
        test_size=0.2,
        random_state=42,
    )
    cancer_pred = cancer_model.predict(cancer_test)
    cancer_accuracy = accuracy_score(y_test, cancer_pred)

    # Diabetes regression performance using R-squared
    diabetes_train, diabetes_test, y_train2, y_test2 = train_test_split(
        diabetes_dataset.data,
        diabetes_dataset.target,
        test_size=0.2,
        random_state=42,
    )
    diabetes_pred = diabetes_model.predict(diabetes_test)
    diabetes_r2 = r2_score(y_test2, diabetes_pred)

    return cancer_accuracy, diabetes_r2


# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
st.sidebar.title("🧠 MedPredict AI")
st.sidebar.caption("AI-powered health screening dashboard")

menu = [
    "Home",
    "Breast Cancer Prediction",
    "Diabetes Prediction",
    "Data Visualization",
    "Model Comparison",
    "About",
]
page = st.sidebar.radio("Navigation", menu, index=0)

# ------------------------------------------------------------
# Home page
# ------------------------------------------------------------
if page == "Home":
    st.markdown("<h1 style='font-size: 2.4rem; margin-bottom: 0.2rem;'>🏥 MedPredict AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: #475569;'>A modern machine learning web application for health risk prediction and data exploration.</p>", unsafe_allow_html=True)

    st.markdown(
        """
        MedPredict AI combines explainable AI workflows with a polished Streamlit dashboard to help users explore disease-related patterns,
        make predictions, and compare model performance in a simple, professional interface.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ✨ Project Highlights")
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            """
            <div class="hero-card">
                <h3>🩺 Breast Cancer Screening</h3>
                <p>Use the trained classifier to estimate whether a tumor sample appears malignant or benign.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            """
            <div class="hero-card">
                <h3>🩸 Diabetes Progression</h3>
                <p>Estimate diabetes progression using a regression model trained on structured medical data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            """
            <div class="hero-card">
                <h3>📊 Insights Dashboard</h3>
                <p>Explore feature distributions and compare the performance of the saved ML models.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🔧 Technologies Used")
    st.markdown("- Python and Streamlit for the interactive web experience")
    st.markdown("- Scikit-learn for model training and evaluation")
    st.markdown("- Plotly for rich, interactive visualizations")
    st.markdown("- Joblib for loading the trained model artifacts from the models folder")

# ------------------------------------------------------------
# Breast cancer prediction page
# ------------------------------------------------------------
elif page == "Breast Cancer Prediction":
    st.header("🩺 Breast Cancer Prediction")
    st.write("Enter the tumor measurements below to receive a prediction from the trained model.")

    cancer_dataset, _ = load_datasets()
    cancer_model, _ = load_saved_models()
    feature_names = cancer_dataset.feature_names

    # Create a compact, professional input layout with three columns.
    values = {}
    cols = st.columns(3)
    for idx, feature in enumerate(feature_names):
        col = cols[idx % 3]
        with col:
            default_value = float(np.mean(cancer_dataset.data[:, idx]))
            values[feature] = st.number_input(
                feature,
                value=default_value,
                format="%.4f",
                key=f"cancer_{idx}",
            )

    if st.button("Predict Breast Cancer", use_container_width=True):
        input_df = pd.DataFrame([values], columns=feature_names)
        prediction = cancer_model.predict(input_df)[0]
        probabilities = cancer_model.predict_proba(input_df)[0]

        label = "Benign" if prediction == 1 else "Malignant"
        confidence = float(np.max(probabilities) * 100)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2f}%")

        prob_df = pd.DataFrame(
            {
                "Class": ["Malignant", "Benign"],
                "Probability": probabilities,
            }
        )
        fig = px.bar(
            prob_df,
            x="Class",
            y="Probability",
            color="Class",
            title="Prediction Probability",
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Diabetes prediction page
# ------------------------------------------------------------
elif page == "Diabetes Prediction":
    st.header("🩸 Diabetes Prediction")
    st.write("Enter the patient health indicators below to estimate diabetes progression.")

    _, diabetes_dataset = load_datasets()
    _, diabetes_model = load_saved_models()
    feature_names = diabetes_dataset.feature_names

    values = {}
    cols = st.columns(3)
    for idx, feature in enumerate(feature_names):
        col = cols[idx % 3]
        with col:
            default_value = float(np.mean(diabetes_dataset.data[:, idx]))
            values[feature] = st.number_input(
                feature,
                value=default_value,
                format="%.4f",
                key=f"diabetes_{idx}",
            )

    if st.button("Predict Diabetes Progression", use_container_width=True):
        input_df = pd.DataFrame([values], columns=feature_names)
        prediction = float(diabetes_model.predict(input_df)[0])

        if prediction > 140:
            risk_label = "High risk"
        elif prediction > 90:
            risk_label = "Moderate risk"
        else:
            risk_label = "Low risk"

        st.success(f"Predicted Diabetes Progression: {prediction:.2f}")
        st.info(f"Risk Level: {risk_label}")

# ------------------------------------------------------------
# Data visualization page
# ------------------------------------------------------------
elif page == "Data Visualization":
    st.header("📊 Data Visualization")
    st.write("Explore the distributions of the medical features used by the models.")

    cancer_dataset, diabetes_dataset = load_datasets()

    # Breast cancer histogram
    cancer_df = pd.DataFrame(cancer_dataset.data, columns=cancer_dataset.feature_names)
    cancer_df["Target"] = cancer_dataset.target
    cancer_df["Target"] = cancer_df["Target"].map({0: "Malignant", 1: "Benign"})

    cancer_feature = st.selectbox(
        "Select a breast cancer feature",
        cancer_dataset.feature_names,
        key="cancer_feature",
    )
    cancer_fig = px.histogram(
        cancer_df,
        x=cancer_feature,
        color="Target",
        nbins=20,
        title=f"Distribution of {cancer_feature} by Tumor Type",
    )
    st.plotly_chart(cancer_fig, use_container_width=True)

    # Diabetes histogram and scatter plot
    diabetes_df = pd.DataFrame(diabetes_dataset.data, columns=diabetes_dataset.feature_names)
    diabetes_df["Target"] = diabetes_dataset.target
    diabetes_feature = st.selectbox(
        "Select a diabetes feature",
        diabetes_dataset.feature_names,
        key="diabetes_feature",
    )

    diabetes_hist = px.histogram(
        diabetes_df,
        x=diabetes_feature,
        nbins=25,
        title=f"Distribution of {diabetes_feature}",
    )
    st.plotly_chart(diabetes_hist, use_container_width=True)

    diabetes_scatter = px.scatter(
        diabetes_df,
        x=diabetes_feature,
        y="Target",
        title=f"{diabetes_feature} vs Diabetes Progression",
        trendline="ols",
    )
    st.plotly_chart(diabetes_scatter, use_container_width=True)

# ------------------------------------------------------------
# Model comparison page
# ------------------------------------------------------------
elif page == "Model Comparison":
    st.header("🤖 Model Comparison")
    st.write("Compare the performance of the saved ML models using evaluation metrics.")

    cancer_accuracy, diabetes_r2 = get_model_metrics()

    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Breast Cancer Model</h3>
                <p><strong>Accuracy:</strong> {cancer_accuracy * 100:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with metric_cols[1]:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Diabetes Model</h3>
                <p><strong>R² Score:</strong> {diabetes_r2:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    comparison_df = pd.DataFrame(
        {
            "Model": ["Breast Cancer Random Forest", "Diabetes Linear Regression"],
            "Metric Value": [cancer_accuracy, diabetes_r2],
            "Metric Type": ["Accuracy", "R² Score"],
        }
    )
    comparison_fig = px.bar(
        comparison_df,
        x="Model",
        y="Metric Value",
        color="Metric Type",
        title="Model Performance Overview",
        labels={"Metric Value": "Metric Value"},
    )
    st.plotly_chart(comparison_fig, use_container_width=True)

# ------------------------------------------------------------
# About page
# ------------------------------------------------------------
else:
    st.header("ℹ️ About MedPredict AI")
    st.markdown(
        """
        MedPredict AI is a lightweight health analytics application that demonstrates how machine learning can be wrapped in a polished
        user experience for disease screening and exploration. The project uses trained models saved in the models folder and presents them
        through a modern Streamlit dashboard.
        """
    )

    st.markdown("### 🧠 Machine Learning Models")
    st.markdown("- Breast Cancer Classification using a Random Forest classifier")
    st.markdown("- Diabetes Progression regression using a Linear Regression model")

    st.markdown("### 🛠️ Technologies")
    st.markdown("- Streamlit for the interactive UI")
    st.markdown("- Plotly for data visualizations")
    st.markdown("- Scikit-learn for model training and evaluation")
    st.markdown("- Joblib for loading saved model artifacts")

    st.markdown("### 🎯 Project Objective")
    st.markdown("The goal of this project is to show how AI can support medical prediction workflows with a professional, easy-to-use web interface.")