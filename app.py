import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Assignment 2 Evaluator", layout="wide")

st.title("Machine Learning Assignment 2 — Classifier Evaluator")
st.markdown("Interactive Streamlit App for Model Evaluation")

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV File", type=["csv"])

model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

selected_model_name = st.sidebar.selectbox("Select ML Model", list(model_options.keys()))
target_column = st.sidebar.text_input("Target Column Name", value="target")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.subheader("Test Dataset Preview")
    st.dataframe(test_df.head())
    
    if target_column not in test_df.columns:
        st.error(f"Target column '{target_column}' not found in uploaded file.")
    else:
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        
        try:
            scaler = joblib.load("model/scaler.pkl")
            model = joblib.load(model_options[selected_model_name])
            
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            st.subheader(f"Evaluation Metrics — {selected_model_name}")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("AUC", f"{auc:.4f}")
            c3.metric("Precision", f"{prec:.4f}")
            c4.metric("Recall", f"{rec:.4f}")
            c5.metric("F1 Score", f"{f1:.4f}")
            c6.metric("MCC", f"{mcc:.4f}")
            
            st.markdown("---")
            col_l, col_r = st.columns(2)
            
            with col_l:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
                
            with col_r:
                st.subheader("Classification Report")
                rep = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(rep).transpose())
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.info("Please upload `test_data.csv` via the sidebar to begin evaluation.")