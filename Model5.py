import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

import shap
import warnings
warnings.filterwarnings("ignore")


csv_path = r"C:\Users\ASUS\Documents\ModelX\Dementia Prediction Dataset_filtered.csv"
df = pd.read_csv(csv_path)


target_col = "NACCUDSD"  # severity out of 5 (multiclass)
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")


df = df[~df[target_col].isna()].copy()


drop_cols = [col for col in ["NACCID", "VISITMO", "VISITDAY", "VISITYR"] if col in df.columns]

X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
y = df[target_col].astype(int) - 1  # ensure integer classes 1..5


numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    remainder="drop"
)


xgb_model = XGBClassifier(
    n_estimators=480,
    max_depth=16,
    learning_rate=0.035,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.03,
    reg_alpha=0.02,
    reg_lambda=1,
    objective="multi:softprob",  
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=45
)


clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", xgb_model)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=45,
    stratify=y 
)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print(f"Accuracy: {acc*100:.2f}%")
print(f"Macro F1: {macro_f1:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))


preprocessor = clf.named_steps["preprocess"]
model = clf.named_steps["model"]

X_test_transformed = preprocessor.transform(X_test)


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_transformed)  


if isinstance(shap_values, list):
    
    shap_sum = np.sum([np.abs(sv) for sv in shap_values], axis=0)
    shap.summary_plot(shap_sum, X_test_transformed, show=False)
else:
    shap.summary_plot(shap_values, X_test_transformed, show=False)

print("SHAP summary plot generated (ensure a plotting backend is available).")