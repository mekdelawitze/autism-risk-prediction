# Autism Risk Prediction from Resting-State fMRI (ABIDE)

A small-scale, portfolio-ready project predicting autism spectrum disorder (ASD)
diagnosis from resting-state functional MRI connectivity, using the public
**ABIDE** (Autism Brain Imaging Data Exchange) preprocessed dataset.

## Goal

Classify ASD vs. typically-developing (TD) subjects from functional connectivity
matrices, with an emphasis on honest validation (site-aware cross-validation)
rather than chasing the highest possible accuracy.

## Why this scope

- Uses **already-preprocessed** data (Preprocessed Connectomes Project output via
  `nilearn`), so no raw MRI preprocessing pipeline needed.
- Classical ML models (logistic regression, SVM, gradient boosting) — appropriate
  for the sample size (~800-1000 subjects after QC) and easy to interpret.
- Deliberately checks for the **site confound** that inflates a lot of published
  ABIDE results — this is the detail that signals real understanding, not just
  running sklearn on a CSV.

## Project structure

```
autism-risk-prediction/
├── data/                  # downloaded ABIDE data lands here (gitignored)
├── notebooks/
│   └── 01_explore.ipynb   # optional: exploratory analysis
├── src/
│   ├── fetch_data.py      # downloads ABIDE phenotypic + connectivity data
│   ├── build_features.py  # builds connectivity feature matrix + labels
│   ├── train.py           # trains + evaluates models (site-stratified CV)
│   └── evaluate.py        # generates ROC curve, confusion matrix, feature importance plots
├── results/                # saved metrics, plots
├── requirements.txt
└── README.md
```

## Setup (run this on a machine with internet access)

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Download ABIDE phenotypic data + precomputed connectivity (~few hundred MB)
python src/fetch_data.py

# 2. Build the feature matrix (connectivity vectors) and labels
python src/build_features.py

# 3. Train models with site-stratified cross-validation
python src/train.py

# 4. Generate evaluation plots (ROC, confusion matrix, top features)
python src/evaluate.py
```

Outputs (metrics, plots) are saved to `results/`.

## Roadmap / what to write up

1. **Baseline**: logistic regression / SVM on CC200 atlas connectivity, standard
   k-fold CV. Report accuracy, ROC-AUC, sensitivity/specificity.
2. **Site-generalization check**: leave-one-site-out CV. Compare performance to
   standard k-fold — the gap is your most interesting finding, and most public
   ABIDE tutorials skip this.
3. **Interpretability**: which connectivity edges (region pairs) are most
   predictive? Relate 2-3 of the top features to known ASD neuroscience findings
   (e.g., default mode network, salience network) in your writeup.
4. **(Optional stretch)** Compare 2 atlases (e.g. CC200 vs Harvard-Oxford) or
   2 harmonization strategies (raw vs. ComBat-corrected).

## Notes on scope / ethics for the writeup

Be explicit in your README/report that this is a research/learning exercise, not
a diagnostic tool — ASD diagnosis is clinical and multi-factorial, and framing
matters both for accuracy of claims and for how "risk prediction" work is
received by the autistic community.

## Dataset citation

Di Martino, A., et al. (2014). The Autism Brain Imaging Data Exchange:
towards a large-scale evaluation of the intrinsic brain architecture in autism.
*Molecular Psychiatry*.
