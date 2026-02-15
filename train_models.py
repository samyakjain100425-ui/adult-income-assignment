# ============================================
# MACHINE LEARNING ASSIGNMENT - MODEL TRAINING
# ============================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ============================================
# CREATE FOLDERS
# ============================================
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================
# LOAD DATA
# ============================================
print("Loading dataset...")

df = pd.read_csv("data/adult.csv")
df.columns = df.columns.str.strip()

# Handle missing values
df = df.replace(" ?", np.nan)
df = df.dropna()


# ============================================
# TARGET VARIABLE
# ============================================
target_col = "income"
df[target_col] = df[target_col].str.strip()

y = df[target_col].map({"<=50K": 0, ">50K": 1})
X = df.drop(target_col, axis=1)


# ============================================
# ENCODING
# ============================================
X = pd.get_dummies(X, drop_first=True)

# Save feature columns
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")


# ============================================
# TRAIN TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================
# SCALING
# ============================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")


# ============================================
# DEFINE MODELS
# ============================================
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(
        n_estimators=100,   # reduce trees
        max_depth=15,       # limit tree depth
        random_state=42
    ),
    "xgboost": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False
    )
}


# ============================================
# TRAIN + EVALUATE
# ============================================
results = []

print("\nTraining models...\n")

for name, model in models.items():

    print(f"Training {name}...")

    # Use scaled data only where required
    if name in ["logistic", "knn", "naive_bayes"]:
        Xtr, Xte = X_train_scaled, X_test_scaled
    else:
        Xtr, Xte = X_train, X_test

    model.fit(Xtr, y_train)

    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    }

    results.append(metrics)

    # Save trained model
    joblib.dump(model, f"models/{name}.pkl")


# ============================================
# SAVE RESULTS
# ============================================
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/metrics.csv", index=False)

print("\nModel Comparison Results:\n")
print(results_df)

print("\nTraining completed successfully!")
