# Step-by-step guide: Customer Churn Prediction

## Step 1 — Extract and open
Extract the ZIP file and open the `customer_churn_prediction` folder in VS Code. Choose Terminal → New Terminal.

## Step 2 — Create a virtual environment
Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

## Step 3 — Install packages
```powershell
python -m pip install -r requirements.txt
```

## Step 4 — Explore the dataset
Open `data/customer_churn.csv`. Each row represents a sample customer. The `Churn` column is the target:
- `Yes`: customer churned
- `No`: customer stayed

The dataset is synthetic and included for learning. It contains features such as tenure, contract type, monthly charges, support contacts, and payment method.

## Step 5 — Train models
```powershell
python train_model.py
```
The script:
1. Loads and cleans the data.
2. Splits features and target.
3. Separates numerical and categorical columns.
4. Imputes missing values, scales numeric values, and one-hot encodes categories.
5. Trains Logistic Regression and Random Forest.
6. Evaluates both models using accuracy, precision, recall, F1-score, a classification report, and confusion matrix.
7. Selects the model with the best F1-score and saves it to `models/churn_model.joblib`.

## Step 6 — Launch the app
```powershell
streamlit run app.py
```
Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Step 7 — Make a prediction
Use the form to enter customer details and select **Predict Customer Churn**. The app displays a churn/stay prediction and, when available, the model-estimated churn probability.

## Step 8 — Understand evaluation
- Accuracy: overall correctness.
- Precision: how often a predicted churn is correct.
- Recall: how many actual churn cases were detected.
- F1-score: harmonic mean of precision and recall.
- Confusion matrix: counts correct and incorrect predictions by class.

Recall can be important when missed churners are costly; precision matters when retention offers are expensive. Choose a metric based on the business objective.

## Common errors
- **Model not found:** run `python train_model.py` before `streamlit run app.py`.
- **CSV not found:** run commands from the project root and confirm `data/customer_churn.csv` exists.
- **Package installation blocked:** use an approved Python environment such as Google Colab or contact your administrator.
