"""
Build functional connectivity feature vectors from ROI timeseries,
and align with labels (ASD=1 / TD=0) and site info.

Saves:
  data/X_features.npy   (n_subjects, n_edges)
  data/y_labels.npy     (n_subjects,)
  data/site_ids.npy     (n_subjects,)  -- needed for leave-one-site-out CV
"""
import os
import pickle
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# --- Load data ---
pheno = pd.read_csv(os.path.join(DATA_DIR, "phenotypic.csv"))
with open(os.path.join(DATA_DIR, "rois_cc200.pkl"), "rb") as f:
    timeseries_list = pickle.load(f)

print(f"Loaded {len(timeseries_list)} subjects' timeseries")

# --- Compute connectivity matrices ---
# Correlation-based functional connectivity is the standard first choice.
# tangent space embedding is a common upgrade if you want to push further.
conn_measure = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
X = conn_measure.fit_transform(timeseries_list)
print(f"Feature matrix shape: {X.shape}  (subjects x edges)")

# --- Labels ---
# ABIDE phenotypic: DX_GROUP == 1 -> Autism, DX_GROUP == 2 -> Control
# Recode so 1 = ASD, 0 = TD (more intuitive for "risk" framing)
y = (pheno["DX_GROUP"] == 1).astype(int).values
print(f"Class balance: {np.bincount(y)} (0=TD, 1=ASD)")

# --- Site IDs (needed for leave-one-site-out CV) ---
site_ids = pheno["SITE_ID"].values

# --- Save ---
np.save(os.path.join(DATA_DIR, "X_features.npy"), X)
np.save(os.path.join(DATA_DIR, "y_labels.npy"), y)
np.save(os.path.join(DATA_DIR, "site_ids.npy"), site_ids)
print(f"Saved features/labels/site_ids to {DATA_DIR}/")
