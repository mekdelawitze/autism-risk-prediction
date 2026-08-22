"""
Download ABIDE phenotypic data and precomputed functional connectivity
(CC200 atlas, filt_global-preprocessed timeseries) via nilearn.

This uses the Preprocessed Connectomes Project (PCP) mirror, so there is
no raw MRI preprocessing to do ourselves.

Run once. Downloads land in ../data/ (a few hundred MB).
"""
import os
from nilearn import datasets

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Subset of sites keeps download small/fast for a first pass.
# Drop `n_subjects` and `pipeline`/`derivatives` args below to get the full dataset.
N_SUBJECTS = 200  # start small; bump up once the pipeline works end-to-end

print(f"Downloading ABIDE data (n_subjects={N_SUBJECTS}) to {DATA_DIR} ...")

abide = datasets.fetch_abide_pcp(
    data_dir=DATA_DIR,
    n_subjects=N_SUBJECTS,
    pipeline="cpac",  # preprocessing pipeline used by PCP
    band_pass_filtering=True,
    global_signal_regression=False,
    derivatives=["rois_cc200"],  # region-of-interest timeseries, CC200 atlas
    quality_checked=True,  # drop subjects that failed QC
)

print("Download complete.")
print(f"Number of subjects: {len(abide.phenotypic)}")
print(f"Phenotypic columns: {list(abide.phenotypic.columns)[:10]} ...")

# Save phenotypic table for later use
import pandas as pd

pheno = pd.DataFrame(abide.phenotypic)
pheno.to_csv(os.path.join(DATA_DIR, "phenotypic.csv"), index=False)
print(f"Saved phenotypic data to {DATA_DIR}/phenotypic.csv")

# rois_cc200 is a list of arrays (timeseries per subject) — pickle for reuse
import pickle

with open(os.path.join(DATA_DIR, "rois_cc200.pkl"), "wb") as f:
    pickle.dump(abide.rois_cc200, f)
print(f"Saved CC200 ROI timeseries to {DATA_DIR}/rois_cc200.pkl")
