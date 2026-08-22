"""
Train baseline classifiers and evaluate with:
  1. Standard stratified k-fold CV
  2. Leave-one-site-out CV (the more honest generalization check)

Saves metrics to results/metrics.json
"""
import os
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X_features.npy"))
y = np.load(os.path.join(DATA_DIR, "y_labels.npy"))
site_ids = np.load(os.path.join(DATA_DIR, "site_ids.npy"), allow_pickle=True)

print(f"X: {X.shape}, y: {y.shape}, sites: {len(set(site_ids))} unique")

models = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    "svm_rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True)),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3),
}

results = {}

# --- 1. Standard stratified k-fold ---
print("\n=== Standard 5-fold stratified CV ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    results[f"{name}_kfold_auc_mean"] = float(scores.mean())
    results[f"{name}_kfold_auc_std"] = float(scores.std())
    print(f"{name}: AUC = {scores.mean():.3f} +/- {scores.std():.3f}")

# --- 2. Leave-one-site-out CV (honest generalization check) ---
print("\n=== Leave-one-site-out CV ===")
logo = LeaveOneGroupOut()
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=logo, groups=site_ids, scoring="roc_auc")
    results[f"{name}_logo_auc_mean"] = float(scores.mean())
    results[f"{name}_logo_auc_std"] = float(scores.std())
    print(f"{name}: AUC = {scores.mean():.3f} +/- {scores.std():.3f}")

# --- Compare the gap ---
print("\n=== Generalization gap (kfold AUC - leave-one-site-out AUC) ===")
for name in models:
    gap = results[f"{name}_kfold_auc_mean"] - results[f"{name}_logo_auc_mean"]
    results[f"{name}_generalization_gap"] = float(gap)
    print(f"{name}: gap = {gap:.3f}")

with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved metrics to {RESULTS_DIR}/metrics.json")
