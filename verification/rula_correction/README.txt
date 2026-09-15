RULA_Left data-entry correction (7.15 -> 7.00) — full pipeline re-verification
================================================================================

Background: two records (Participant ID 10014 and 10045) had RULA_Left = 7.15
in the analysis dataset, exceeding the theoretical grand-score maximum of 7.
The author confirmed this is a data-entry error and the correct value is 7.00
for both records.

Files:
- WMSD_analysis_deidentified_corrected.xlsx : the analysis dataset with
  RULA_Left corrected to 7.00 for the two affected records; all other values
  unchanged.
- model_summary_primary_ORIGINAL_data.csv / _CORRECTED_data.csv : full output
  of src/run_reanalysis.py (unmodified nested-CV pipeline, 10 repeats x 5-fold
  outer CV, 3-fold inner tuning) run independently on the original vs.
  corrected dataset. The ORIGINAL-data run was checked to reproduce the
  cached results/primary/model_summary_primary.csv bit-for-bit, confirming
  the rerun environment is faithful to the one that produced the manuscript's
  reported numbers.
- feature_ablation_summary_ORIGINAL_data.csv / _CORRECTED_data.csv : same
  comparison for the prespecified postural-only and nonpostural-only feature
  sets.
- RULA_correction_impact_comparison.csv : side-by-side merge of the above.

Result: GBT on the combined feature set (the manuscript's reported/selected
model) produced bit-for-bit identical out-of-fold class predictions and
identical balanced accuracy (0.940346), macro-F1 (0.950886), and MCC
(0.905949) on both datasets. The nonpostural ablation is unaffected (it does
not use RULA). The postural-only ablation (RULA_Mean + RULA_Asymmetry alone)
shifts by +0.0018 balanced accuracy / +0.0030 macro-F1 / +0.0056 MCC -- a
third/fourth-decimal change with no effect on its already-reported conclusion
that postural features alone are insufficient. Other classifiers (RF, SVM,
LR, kNN) on the combined set shift by a few thousandths; none change rank
order or the selection of GBT as best model.

Descriptive statistics also re-verified directly from the corrected dataset:
- Left-side weighted RULA mean: 5.022 -> 5.017; SD: 1.120 -> 1.111
- Left-side RULA distribution bins (<=3.00, >3.00-4.00, >4.00-5.00,
  >5.00-6.00, >6.00): unchanged (n = 2, 12, 18, 15, 13) -- both corrected
  records remain in the >6.00 bin.
- Pearson r (left-side RULA vs. MSD Index, from the raw pre-deidentification
  dataset): 0.7477 -> 0.7494, both rounding to the manuscript's reported 0.75.

Note on the environment: this rerun required a one-line, verification-only
patch to run_reanalysis.py's categorical-column detection (it checked
`df[c].dtype == "object"`, which misses pandas' newer string-extension dtype
returned by this environment's pandas 3.0.2 when reading the .xlsx). This is
a pandas-version portability issue in the script, unrelated to the RULA
correction; it only affects re-execution in an environment with this pandas
version, not the correctness of the previously reported cached results.

Scope clarification: these comparison files cover the primary model summaries
and feature ablation only. The complete corrected package also reran the
secondary sensitivity analyses, learning curve, consensus predictions and
Holm-adjusted McNemar comparisons. Those current outputs are authoritative in
results/primary, results/secondary and results/learning_curve. In particular,
the corrected consensus predictions give GBT versus k-NN an unadjusted exact
McNemar p-value of 0.003906 and a Holm-adjusted p-value of 0.027344; the other
six adjusted comparisons are not significant.
