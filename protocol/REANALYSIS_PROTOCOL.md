# WMSD Reviewer-Compliant Reanalysis Protocol

Status: FROZEN BEFORE CORRECTED MODEL RESULTS

## 1. Authoritative data

- The sole modelling dataset is `MSDs Risk - Data.xlsx` with 60 participants.
- Class counts are Low = 6, Medium = 37, and High = 17.
- The 90-participant workbook is obsolete and excluded.
- Participant name is never used or exported.
- `MSDs Index` and the nine pain indicators are excluded from predictors because the risk-class target is derived from symptom/SEP information.

## 2. Prediction target and analysis unit

- Target: current WMSD risk class (`Low`, `Medium`, `High`).
- Unit of analysis and resampling: participant.
- This is internal validation of a screening classifier, not clinical diagnosis or prospective disease prediction.

## 3. Prespecified features

To avoid outcome-informed feature-selection leakage, logistic-regression significance filtering is removed from the predictive pipeline.

Postural features:

- Mean bilateral time-weighted RULA: `(RULA_Left + RULA_Right) / 2`.
- Bilateral RULA asymmetry: `abs(RULA_Left - RULA_Right)`.

Non-postural features:

- Age, sex, BMI, medical history, smoking, alcohol use, physical exercise, job tenure, working days, daily work duration, daily break duration, table height, and sitting height.
- Height and weight are not entered with BMI to reduce deterministic/redundant anthropometric information.

The primary feature set is the combined postural and non-postural set. Postural-only and non-postural-only sets are evaluated as prespecified ablations.

## 4. Preprocessing

- Blank medical-history, alcohol, and exercise entries are explicitly coded as `None reported`, consistent with the manuscript summaries.
- Numeric variables are median-imputed and standardized within each training partition.
- Categorical variables are most-frequent-imputed and one-hot encoded within each training partition.
- Unknown test-fold categories are ignored during transformation.

## 5. Validation

- Outer validation: 10 repeats of stratified five-fold cross-validation (50 held-out folds).
- Inner tuning: stratified three-fold cross-validation within each outer-training fold.
- Seeds: outer repeats generated from seed 20260904; inner-fold seeds are deterministically derived from outer repeat/fold identifiers.
- Every preprocessing and tuning operation is fitted only with the relevant training partition.
- Each participant receives exactly one held-out prediction per repeat.

## 6. Models

Eight prespecified classifiers are retained: multinomial logistic regression, Gaussian naive Bayes, k-nearest neighbours, decision tree, random forest, support-vector machine, AdaBoost, and gradient-boosted trees. Hyperparameter grids are deliberately small because n = 60.

## 7. Model selection and metrics

- Inner-loop selection metric: balanced accuracy.
- Primary reported metrics: balanced accuracy, macro-F1, and Matthews correlation coefficient.
- Secondary metrics: overall accuracy, macro precision, macro recall, and class-specific sensitivity, specificity, precision, and F1.
- Regression metrics (R-squared, MAE, MSE, RMSE) are prohibited.
- Primary summaries use the distribution of the 10 repeat-level held-out estimates.
- A consensus confusion matrix is constructed by averaging each participant's held-out class probabilities across repeats and taking the maximum-probability class.

## 8. Paired comparisons

- The best model is defined by mean repeat-level balanced accuracy.
- Pairwise comparisons use one consensus held-out prediction per participant.
- Exact paired McNemar tests compare correctness discordance between the best model and every competitor.
- Holm correction controls multiplicity across the seven comparisons.

## 9. Sensitivity and ablation analyses

- Feature ablation: postural only, non-postural only, and combined features for the best-performing algorithm.
- Class-imbalance sensitivity: unweighted versus class-weighted fitting when the selected algorithm supports class weights. If not supported, the limitation is reported and no synthetic resampling is introduced post hoc.
- SMOTE is not used in the primary corrected analysis because the smallest inner-training partitions contain only two to four Low-risk participants; synthetic interpolation would be unstable. The original SMOTE result is treated as an optimistic legacy analysis, not confirmatory evidence.

## 10. Remaining separate reviewer analyses

- Missing-frame sensitivity cannot be generalized from one worker; supplied Kinovea files are used only to explain the tracking workflow and illustrate missingness.
- End-to-end timing must be run on the authors' Windows 11 computer (10th-generation Core i5, 8 GB RAM, NVIDIA GTX 1650). A benchmark script will be supplied.

