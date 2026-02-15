import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef
)

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Income Prediction Dashboard",
    layout="wide"
)

# ==============================
# CUSTOM HEADER
# ==============================
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
    💼 Income Prediction Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "### Compare Multiple Machine Learning Models on Test Data"
)

st.markdown("---")

# ==============================
# LOAD MODELS
# ==============================
MODEL_FOLDER = "models"

available_models = [
    f.replace(".pkl", "")
    for f in os.listdir(MODEL_FOLDER)
    if f.endswith(".pkl") and f not in ["scaler.pkl", "feature_columns.pkl"]
]

# ==============================
# SIDEBAR
# ==============================
st.sidebar.markdown("## ⚙ Model Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    available_models
)

# ==============================
# MAIN LOGIC
# ==============================
if uploaded_file is not None:

    # Load artifacts
    model = joblib.load(f"{MODEL_FOLDER}/{selected_model_name}.pkl")
    scaler = joblib.load(f"{MODEL_FOLDER}/scaler.pkl")
    feature_columns = joblib.load(f"{MODEL_FOLDER}/feature_columns.pkl")

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    if "income" not in df.columns:
        st.error("Uploaded CSV must contain 'income' column.")
    else:

        df["income"] = df["income"].str.strip()
        y_true = df["income"].map({"<=50K": 0, ">50K": 1})

        X = df.drop("income", axis=1)
        X = pd.get_dummies(X, drop_first=True)

        # Align with training columns
        X = X.reindex(columns=feature_columns, fill_value=0)

        # Apply scaling if needed
        if selected_model_name in ["logistic", "knn", "naive_bayes"]:
            X = scaler.transform(X)

        # Predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # ==============================
        # METRICS SECTION
        # ==============================
        st.subheader("📊 Model Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
        col2.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
        col3.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
        col4.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
        col5.metric("AUC Score", f"{roc_auc_score(y_true, y_prob):.4f}")
        col6.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.4f}")

        st.markdown("---")

        # ==============================
        # CONFUSION MATRIX
        # ==============================
        st.subheader("🔎 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["<=50K", ">50K"],
            yticklabels=["<=50K", ">50K"]
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        st.pyplot(fig)

        st.markdown("---")

        # ==============================
        # CLASSIFICATION REPORT
        # ==============================
        st.subheader("📑 Classification Report")

        report = classification_report(
            y_true,
            y_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        st.markdown("---")

        # ==============================
        # PREDICTION PREVIEW
        # ==============================
        st.subheader("📌 Prediction Preview")

        preview_df = df.copy()
        preview_df["Predicted Income"] = y_pred
        preview_df["Predicted Income"] = preview_df["Predicted Income"].map(
            {0: "<=50K", 1: ">50K"}
        )

        st.dataframe(preview_df.head(10))

else:
    st.info("Please upload a test CSV file from the sidebar to begin evaluation.")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown(
    "<center>Developed by 2025AA05245 SAMYAK JAIN Machine Learning Assignment 2 | 2026</center>",
    unsafe_allow_html=True
)
