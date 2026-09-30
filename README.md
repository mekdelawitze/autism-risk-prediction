# Autism Classification from Resting-State fMRI

Comparing logistic regression, SVM, and gradient boosting for ASD versus typically developing classification using functional connectivity from the ABIDE dataset.

## Overview

I built this project to examine how classification performance changes when a model is evaluated on an acquisition site that was not included in training. I compared stratified cross-validation with leave-one-site-out (LOSO) cross-validation using the same connectivity features and model configurations.

## Data and methods

I used ABIDE data accessed through Nilearn and the Preprocessed Connectomes Project. The analysis includes the first 200 eligible participants returned after quality filtering: 111 ASD and 89 typically developing participants across six sites, with ages ranging from 8.0 to 35.2 years.

| Site | ASD | TD |
|---|---:|---:|
| OHSU | 12 | 13 |
| OLIN | 14 | 14 |
| PITT | 24 | 26 |
| SDSU | 8 | 19 |
| TRINITY | 19 | 14 |
| UM_1 | 34 | 3 |

The downloaded data are C-PAC-preprocessed regional time series with band-pass filtering enabled and global signal regression disabled. Raw MRI preprocessing was not performed in this repository.

- **Features:** correlation connectivity calculated with Nilearn's `ConnectivityMeasure`, using its default shrinkage covariance estimator.
- **Parcellation:** 200 CC200 regions, giving 19,900 unique region-pair features per participant.
- **Models:** L2-regularized logistic regression, RBF-kernel SVM, and gradient boosting with 200 estimators and maximum tree depth 3.
- **Scaling:** standardization within each training fold for logistic regression and SVM.
- **Evaluation:** five-fold stratified cross-validation and LOSO cross-validation.

Model hyperparameters were fixed rather than selected through a tuning search. The stratified splits and gradient boosting model use `random_state=42`.

## Results

| Model | Stratified CV AUC | LOSO AUC | Difference |
|---|---:|---:|---:|
| Logistic regression | 0.747 ± 0.037 | 0.656 ± 0.059 | 0.092 |
| SVM, RBF kernel | 0.734 ± 0.055 | 0.605 ± 0.138 | 0.129 |
| Gradient boosting | 0.740 ± 0.057 | 0.634 ± 0.081 | 0.106 |

Values are mean AUC ± standard deviation across folds, not confidence intervals. Differences were calculated before rounding.

All three models had lower average AUC under LOSO evaluation. Logistic regression had the highest observed mean AUC and lowest fold-to-fold standard deviation under both schemes. These results describe the tested configurations and do not establish that one model is generally superior.

The difference between evaluation schemes suggests that performance depends on whether the acquisition site is represented during training. This experiment does not isolate the contributions of scanner differences, participant characteristics, or training-set size.

### Single-split evaluation

I also evaluated logistic regression on a stratified 75/25 split to generate a ROC curve and confusion matrix. This split produced an AUC of 0.646. It uses the same participant pool as the cross-validation analysis and is not an independent external test.

![Logistic regression ROC curve from a stratified 75/25 split](results/roc_curve.png)

![Logistic regression confusion matrix from a stratified 75/25 split](results/confusion_matrix.png)

## Connectivity feature analysis

I ranked the 20 largest absolute logistic-regression coefficients from the model fitted on the 75% training split and mapped them to pairs of CC200 time-series columns.

Twelve coefficients were negative, and region index 191 appeared in three of the selected pairs. A negative coefficient means that lower standardized connectivity contributes toward the ASD prediction, conditional on the other features in the model.

These coefficients do not establish significant group differences or identify a biological hub. Their stability across training splits has not yet been evaluated.

Region numbers are zero-based column indices. Their correspondence to atlas-image labels or anatomical regions has not been verified.

![Largest-magnitude logistic regression coefficients](results/top_features.png)

The coefficient table is saved in [results/top_edges_regions.csv](results/top_edges_regions.csv).

## Limitations

The cohort is a convenience subset rather than a randomly selected or site-balanced sample. Class balance differs between sites, particularly at UM_1, which includes only three typically developing participants.

The feature count is large relative to the sample size. Regularization and constrained tree depth limit model complexity, but do not remove the risk of unstable estimates.

Age, sex, and head motion were not explicitly adjusted for in this analysis. Quality filtering does not establish that these potential confounds have been removed.

The evaluation schemes differ in training-set size and aggregation: stratified CV averages five fold scores, while LOSO averages six site scores with equal weight per site. The reported difference is therefore not a direct estimate of scanner effects alone.

This analysis concerns classification within an existing dataset, not prediction of future autism risk or clinical diagnosis.

## Implementation notes

I fixed the gradient boosting random seed and updated its saved metrics and result table. I also clarified the region-index descriptions in the mapping script and pinned package versions in `requirements.txt`.

The evaluation currently saves aggregate metrics. Saving participant identifiers, fold assignments, per-site scores, and held-out predictions would make the results easier to inspect and reproduce.

## Next steps

I would first add explicit checks for label validity, feature quality, and alignment between participant records and time series.

Further analysis would examine performance by site, demographic and motion-related confounding, and coefficient stability across folds. A larger, explicitly defined cohort would also help assess how well these results hold beyond the current subset.

## Running the project

From the repository root:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the regional time series and build connectivity features:

```bash
python src/fetch_data.py
python src/build_features.py
```

Run the cross-validation comparisons:

```bash
python src/train.py
```

Generate the single-split figures and coefficient table:

```bash
python src/evaluate.py
python src/edge_to_regions.py
```

Outputs are saved in `results/`. Package versions are pinned, but identical results across environments have not been verified.

## References

- [ABIDE](https://fcon_1000.projects.nitrc.org/indi/abide/)
- [Preprocessed Connectomes Project: ABIDE](https://preprocessed-connectomes-project.org/abide/)
- [Nilearn](https://nilearn.github.io/)
- [scikit-learn](https://scikit-learn.org/)