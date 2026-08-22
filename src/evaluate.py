"""
Fit the best model on a train/test split and generate:
  - ROC curve
  - Confusion matrix
  - Top predictive connectivity edges (feature importance)

Saves plots to results/
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X_features.npy"))
y = np.load(os.path.join(DATA_DIR, "y_labels.npy"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# --- ROC curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — ASD vs TD Classification")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
plt.close()

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["TD", "ASD"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# --- Top predictive features (logistic regression coefficients) ---
clf = model.named_steps["logisticregression"]
coefs = clf.coef_[0]
top_idx = np.argsort(np.abs(coefs))[::-1][:20]

plt.figure(figsize=(8, 6))
sns.barplot(x=coefs[top_idx], y=[f"edge_{i}" for i in top_idx], orient="h")
plt.xlabel("Logistic Regression Coefficient")
plt.title("Top 20 Most Predictive Connectivity Edges")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "top_features.png"), dpi=150)
plt.close()

print(f"Test set AUC: {roc_auc:.3f}")
print(f"Saved plots to {RESULTS_DIR}/")
print("\nNote: 'edge_N' indices map back to ROI pairs in the CC200 atlas —")
print("use nilearn's atlas labels to translate these into named brain regions")
print("for your writeup (e.g. 'default mode network - salience network').")
