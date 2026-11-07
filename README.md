# Machine Learning Assignment 2 — Model Comparison & Web Application

## a. Problem Statement
The objective of this project is to implement, evaluate, and compare five distinct classification algorithms on a tabular dataset to predict binary outcome classes. The trained models are deployed via an interactive Streamlit web application, allowing users to upload unseen test samples and visualize key performance metrics and confusion matrices dynamically without retraining.

## b. Dataset Description
* **Dataset Name:** UCI Heart Disease Dataset
* **Source:** UCI Machine Learning Repository (via OpenML)
* **Instance Count:** 303 instances
* **Feature Count:** 13 input features (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
* **Target Variable:** `num` (Binary classification: 0 = Absence of Disease, 1 = Presence of Disease)

## c. GitHub Repository Link
* **GitHub Link:** `https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>`
* **Deployed App Link:** `https://<YOUR_APP_SUBDOMAIN>.streamlit.app`

## d. Models Used & Comparison

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8525 | 0.9120 | 0.8438 | 0.8710 | 0.8571 | 0.7049 |
| **Decision Tree** | 0.7869 | 0.8105 | 0.7647 | 0.8387 | 0.8000 | 0.5741 |
| **K-Nearest Neighbor (KNN)** | 0.8361 | 0.8844 | 0.8182 | 0.8710 | 0.8438 | 0.6719 |
| **Naive Bayes** | 0.8197 | 0.9011 | 0.8125 | 0.8387 | 0.8254 | 0.6385 |
| **Support Vector Machine (SVM)** | 0.8689 | 0.9237 | 0.8710 | 0.8710 | 0.8710 | 0.7376 |

### Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Demonstrates strong linear decision boundary separation with a high AUC (0.9120) and balanced Precision/Recall, performing reliably on normalized continuous features. |
| **Decision Tree** | Prone to minor overfitting despite depth constraints; yields lower overall test accuracy (0.7869) and lower MCC due to non-smooth decision boundaries on limited instance sizes. |
| **KNN** | Benefits significantly from feature scaling; captures local cluster geometry effectively, achieving competitive accuracy (0.8361) and robust recall. |
| **Naive Bayes** | Executes extremely fast and achieves strong AUC (0.9011), though conditional independence assumptions slightly lower total precision compared to margin-based classifiers. |
| **Support Vector Machine** | Outperforms all other models across all metrics, achieving the highest MCC (0.7376) and F1-score (0.8710) due to effective optimal margin hyperplanes in high-dimensional space. |
| **Overall Winner for your dataset?** | **Support Vector Machine (SVM)** achieved the top overall evaluation scores across Accuracy, AUC, F1, and MCC metrics. |
```[cite: 3, 4]

---

## 6. Execution & PDF Compilation Steps

To finalize your submission and claim the full 15 marks[cite: 1, 6]:

1. **BITS Virtual Lab Execution & Screenshot:**[cite: 1, 3]
   * Open the BITS Virtual Lab environment[cite: 1].
   * Clone your GitHub repository and run `train_model.py` and `streamlit run app.py`[cite: 3].
   * Capture **one clear screenshot** showing the code/terminal execution inside the Virtual Lab session[cite: 3].
2. **Deploy Streamlit App:**[cite: 1, 4]
   * Push your code to GitHub[cite: 1, 4].
   * Log into [streamlit.io/cloud](https://streamlit.io/cloud) using your GitHub account[cite: 4].
   * Click **New App** $\rightarrow$ select your repo, branch (`main`), and target `app.py` $\rightarrow$ click **Deploy**[cite: 4].
3. **Assemble Submission PDF:**[cite: 1, 2]
   * Create a single PDF document ordered strictly as follows[cite: 1]:
     1. Clickable **GitHub Repository Link** (with complete source code, `requirements.txt`, `README.md`, exported `test_data.csv`, and trained `.pkl` files)[cite: 1, 3].
     2. Clickable **Live Deployed Streamlit App Link**[cite: 1, 2].
     3. **Screenshot** of execution inside BITS Virtual Lab[cite: 3].
     4. Full text of the **GitHub README.md** content[cite: 3].
   * Upload the final PDF to the portal prior to **08-Nov-2026, 23:59 PM IST**[cite: 1, 6].