# Customer Churn Prediction

A beginner-friendly machine learning project that predicts whether a customer may leave a service using historical customer data.

## Features
- Data inspection and preprocessing
- Logistic Regression and Random Forest classification
- Missing-value imputation and categorical encoding
- Accuracy, precision, recall, F1-score, classification report, and confusion matrix
- Best model saved using joblib
- Streamlit app for interactive single-customer predictions
- Included sample CSV dataset

## Project structure
```text
customer_churn_prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── STEP_BY_STEP.md
├── README.md
├── data/
│   └── customer_churn.csv
└── models/                 # created after training
```

## Run on Windows / VS Code
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Run training before starting the dashboard. The trained model and metrics are saved in `models/`.

## Dataset
The included educational sample dataset contains customer tenure, service plan, monthly charges, support contacts, contract type, payment method, and churn label. It is synthetic and intended to demonstrate the workflow, not to represent real customers. Replace it with an appropriately licensed real dataset for meaningful business evaluation.

## Metrics
- **Accuracy:** overall fraction of correct predictions.
- **Precision:** among customers predicted to churn, fraction that actually churned.
- **Recall:** among customers who churned, fraction detected by the model.
- **F1-score:** balance between precision and recall.

Churn models can be affected by class imbalance and data leakage. Evaluate on representative held-out data before using predictions in business decisions.
