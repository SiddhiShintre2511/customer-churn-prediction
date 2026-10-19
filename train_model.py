"""Train and evaluate a customer churn classifier; save reusable model artifacts."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "customer_churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    target = "Churn"
    if target not in df.columns:
        raise ValueError(f"Dataset must contain target column '{target}'.")
    df = df.drop_duplicates()
    df[target] = df[target].astype(str).str.strip().str.lower().map({
        "yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0
    })
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    # IDs are identifiers, not useful predictive features.
    X = X.drop(columns=[c for c in ["CustomerID", "customerID", "customer_id"] if c in X.columns], errors="ignore")
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical = X.select_dtypes(include=["number"]).columns.tolist()
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipeline, numerical),
        ("cat", categorical_pipeline, categorical)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    }
    results = {}
    best_name, best_pipeline, best_f1 = None, None, -1
    for name, classifier in candidates.items():
        pipeline = Pipeline([("preprocess", preprocess), ("classifier", classifier)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4)
        }
        results[name] = metrics
        print(f"\n=== {name} ===")
        print(metrics)
        print(classification_report(y_test, pred, target_names=["Stay", "Churn"], zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))
        if metrics["f1_score"] > best_f1:
            best_name, best_pipeline, best_f1 = name, pipeline, metrics["f1_score"]
    joblib.dump(best_pipeline, MODEL_DIR / "churn_model.joblib")
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"selected_model": best_name, "test_metrics": results}, f, indent=2)
    print(f"\nSelected model: {best_name}")
    print(f"Saved model to: {MODEL_DIR / 'churn_model.joblib'}")
    print(f"Saved metrics to: {MODEL_DIR / 'metrics.json'}")

if __name__ == "__main__":
    main()
