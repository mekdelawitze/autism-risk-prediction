"""
Maps the top predictive connectivity "edges" (from evaluate.py's feature
importance plot) back to which pairs of brain regions they represent.

The CC200 atlas used here is made of 200 DATA-DRIVEN clusters (grouped by
similar activity, not classic anatomy) -- so we identify them by their
index number, not a name like "amygdala." This is the honest, correct way
to describe CC200 regions.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

X = np.load(os.path.join(DATA_DIR, "X_features.npy"))
y = np.load(os.path.join(DATA_DIR, "y_labels.npy"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
model.fit(X_train, y_train)
coefs = model.named_steps["logisticregression"].coef_[0]

N_REGIONS = 200
top_n = 20
top_idx = np.argsort(np.abs(coefs))[::-1][:top_n]

# Reconstruct which region-pair each vector index corresponds to.
# nilearn's ConnectivityMeasure(vectorize=True, discard_diagonal=True) flattens
# the lower triangle (excluding the diagonal) in this exact order.
row_idx, col_idx = np.tril_indices(N_REGIONS, k=-1)

print(f"\nTop {top_n} most predictive region-pairs:\n")
rows = []
for rank, idx in enumerate(top_idx, start=1):
    region_a = row_idx[idx]
    region_b = col_idx[idx]
    direction = "higher connectivity -> more ASD-like" if coefs[idx] > 0 else "lower connectivity -> more ASD-like"
    print(f"{rank}. Region {region_a} <-> Region {region_b}  (coef={coefs[idx]:.4f}, {direction})")
    rows.append({"rank": rank, "region_a": region_a, "region_b": region_b,
                 "coefficient": coefs[idx], "direction": direction})

out_df = pd.DataFrame(rows)
out_path = os.path.join(os.path.dirname(__file__), "..", "results", "top_edges_regions.csv")
out_df.to_csv(out_path, index=False)
print(f"\nSaved table to {out_path}")
print("\nNote: 'Region N' numbers refer to CC200 cluster IDs, not anatomical names.")
