import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import plotly.express as px
import numpy as np

# --- Helper Function ---
def plot_confusion_matrix(cm, labels, model_name):
    """Creates a heatmap for confusion matrix."""
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True"),
        x=[f"Class {l}" for l in labels],
        y=[f"Class {l}" for l in labels],
    )
    fig.update_layout(title_text=f"Confusion Matrix: {model_name}", title_x=0.5)
    return fig

# --- Model Runner Function ---
def run_analysis(dataset_choice, user_file, target_column, selected_models):
    df = None
    summary_output = ""
    plots = {}

    # Load dataset
    if dataset_choice == "Iris":
        data = load_iris(as_frame=True)
        df = pd.concat([data.data, data.target.rename("target")], axis=1)
        target_column = "target"
    elif dataset_choice == "Wine":
        data = load_wine(as_frame=True)
        df = pd.concat([data.data, data.target.rename("target")], axis=1)
        target_column = "target"
    elif dataset_choice == "Breast Cancer":
        data = load_breast_cancer(as_frame=True)
        df = pd.concat([data.data, data.target.rename("target")], axis=1)
        target_column = "target"
    elif dataset_choice == "Upload your own":
        if user_file is None:
            st.warning("Please upload a CSV file.")
            return None, None, None, None
        df = pd.read_csv(user_file)

    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in dataset.")
        return None, None, None, None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model dictionary
    model_map = {
        "Passive-Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
        "SVM": SVC(gamma="auto"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    performance_data = []

    # Train and evaluate models
    for name in selected_models:
        model = model_map.get(name)
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            performance_data.append(
                {
                    "Model": name,
                    "Accuracy": f"{accuracy:.4f}",
                    "Precision": f"{precision:.4f}",
                    "Recall": f"{recall:.4f}",
                    "F1-Score": f"{f1:.4f}",
                }
            )

            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
            fig = plot_confusion_matrix(cm, np.unique(y), name)
            plots[name] = fig

        except Exception as e:
            st.error(f"Error training {name}: {e}")

    if not performance_data:
        summary_output = "No models selected."
    else:
        best_model = max(performance_data, key=lambda x: float(x["F1-Score"]))
        summary_output = f"🏆 **Best Model:** {best_model['Model']} (F1 = {best_model['F1-Score']})"

    return df.head(), pd.DataFrame(performance_data), plots, summary_output

# --- Streamlit App Layout ---
st.set_page_config(page_title="ML Model Comparison", layout="wide")
st.title("✨ Machine Learning Model Comparison Tool")
st.markdown("Easily compare multiple ML models on standard or custom datasets.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
dataset_choice = st.sidebar.radio(
    "Select Dataset",
    ["Iris", "Wine", "Breast Cancer", "Upload your own"],
    index=0,
)

user_file = None
if dataset_choice == "Upload your own":
    user_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

target_column = st.sidebar.text_input("Target Column Name", value="target")

selected_models = st.sidebar.multiselect(
    "Select Models",
    ["Passive-Aggressive", "SVM", "Random Forest", "Logistic Regression"],
    default=["Passive-Aggressive", "SVM", "Random Forest", "Logistic Regression"],
)

if st.sidebar.button("🚀 Run Analysis"):
    with st.spinner("Training models... Please wait."):
        preview, metrics_df, plots, summary = run_analysis(
            dataset_choice, user_file, target_column, selected_models
        )

    if preview is not None:
        st.success(summary)

        with st.expander("📊 Dataset Preview"):
            st.dataframe(preview)

        with st.expander("📈 Model Performance Metrics", expanded=True):
            st.dataframe(metrics_df)

        with st.expander("🧩 Confusion Matrices", expanded=True):
            cols = st.columns(2)
            for i, (name, fig) in enumerate(plots.items()):
                if fig is not None:
                    cols[i % 2].plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 Configure your options in the sidebar and click **Run Analysis**.")
