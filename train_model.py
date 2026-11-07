import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def build_and_evaluate():
    os.makedirs("model", exist_ok=True)
    
    # Load Breast Cancer dataset: 30 features, 569 instances (Satisfies >=12 features & >=500 instances)
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    
    target_col = 'target'
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, "model/scaler.pkl")
    
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df[target_col] = y_test.values
    test_df.to_csv("test_data.csv", index=False)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        file_name = f"model/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, file_name)
        print(f"Trained and saved: {name}")
        
    print("Training complete! Artifacts saved successfully.")

if __name__ == "__main__":
    build_and_evaluate()