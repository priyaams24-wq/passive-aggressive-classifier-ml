import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.datasets import load_iris, load_wine
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import label_binarize

# --- Helper Functions ---

def load_dataset_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        data = response.text
        # Replace missing values represented by '?' with NaN
        data = data.replace('?', 'NaN')
        # Load the dataset into a pandas DataFrame
        df = pd.read_csv(StringIO(data), header=None)
        # Assign column names based on the dataset's attribute information
        df.columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error loading dataset: {e}")
        return None


def plot_confusion_matrix(cm, labels, model_name):
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

def plot_feature_importance(model, X, model_name):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        fig = go.Figure([go.Bar(x=X.columns, y=importance)])
        fig.update_layout(title=f"Feature Importance: {model_name}", xaxis_title="Features", yaxis_title="Importance")
        return fig
    return None

def plot_metrics_bar(metrics_df):
    fig = px.bar(metrics_df.melt(id_vars="Model"), 
                 x="Model", y="value", color="variable", barmode="group",
                 title="Model Performance Metrics Comparison")
    return fig

def plot_roc_curve(model, X_test, y_test, model_name):
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
        y_score = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        fig = px.area(
            x=fpr, y=tpr,
            title=f"ROC Curve: {model_name} (AUC={roc_auc:.2f})",
            labels={"x":"False Positive Rate", "y":"True Positive Rate"}
        )
        return fig
    return None

# --- Main Analysis Function ---
def run_analysis(df, target_column, selected_models):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_map = {
        "Passive-Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, gamma="auto"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    performance_data = []
    plots = {}
    feature_plots = {}
    roc_plots = {}

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
                    "Accuracy": round(accuracy, 4),
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1-Score": round(f1, 4),
                }
            )

            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
            plots[name] = plot_confusion_matrix(cm, np.unique(y), name)

            if name == "Random Forest":
                feature_plots[name] = plot_feature_importance(model, X, name)

            if len(np.unique(y)) == 2:
                roc_fig = plot_roc_curve(model, X_test, y_test, name)
                if roc_fig:
                    roc_plots[name] = roc_fig

        except Exception as e:
            st.error(f"Error training {name}: {e}")

    best_model = max(performance_data, key=lambda x: x["F1-Score"])
    summary_output = f"🏆 **Best Model:** {best_model['Model']} (F1 = {best_model['F1-Score']})"

    metrics_df = pd.DataFrame(performance_data)
    metrics_bar_fig = plot_metrics_bar(metrics_df)

    return metrics_df, plots, feature_plots, roc_plots, summary_output, metrics_bar_fig

# --- Streamlit Layout ---
st.set_page_config(page_title="ML Model Comparison", layout="wide")
st.title("✨ Machine Learning Model Comparison Tool")
st.markdown("Compare multiple classification models with rich visualizations.")

# Sidebar
st.sidebar.header("⚙️ Configuration")
dataset_choice = st.sidebar.radio(
    "Select Dataset",
    ["Iris", "Wine", "Heart Disease"]
)

if dataset_choice == "Iris":
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    target_column = "target"
elif dataset_choice == "Wine":
    from sklearn.datasets import load_wine
    data = load_wine(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    target_column = "target"
elif dataset_choice == "Heart Disease":
    heart_url = "https://raw.githubusercontent.com/rahulmidha/Heart-Disease-UCI-Dataset/main/heart.csv"
    df = load_dataset_from_url(heart_url)
    target_column = "target"

selected_models = st.sidebar.multiselect(
    "Select Models",
    ["Passive-Aggressive", "SVM", "Random Forest", "Logistic Regression"],
    default=["Passive-Aggressive", "SVM", "Random Forest", "Logistic Regression"],
)

if st.sidebar.button("🚀 Run Analysis"):
    metrics_df, plots, feature_plots, roc_plots, summary, metrics_bar_fig = run_analysis(df, target_column, selected_models)
    
    st.success(summary)

    with st.expander("📊 Dataset Preview"):
        st.dataframe(df.head())

    with st.expander("📈 Model Performance Metrics", expanded=True):
        st.dataframe(metrics_df)
        st.plotly_chart(metrics_bar_fig, use_container_width=True)

    with st.expander("🧩 Confusion Matrices", expanded=True):
        cols = st.columns(2)
        for i, (name, fig) in enumerate(plots.items()):
            if fig:
                cols[i % 2].plotly_chart(fig, use_container_width=True)

    with st.expander("🌟 Feature Importance (Random Forest)"):
        for fig in feature_plots.values():
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    if roc_plots:
        with st.expander("📊 ROC Curves (Binary Classification)", expanded=True):
            for fig in roc_plots.values():
                st.plotly_chart(fig, use_container_width=True)
