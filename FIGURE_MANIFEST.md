# Figure manifest

All regenerated scientific figures use a Times New Roman-compatible serif stack and are exported at 600 dpi. The PNG files in `figures/manuscript/` are the manuscript-ready versions.

| Figure | File | Reproducible code/source | Editable diagram source |
|---|---|---|---|
| 1 | `Figure_01_workflow.png` | `src/generate_study_workflow_figures.py` | `figures/diagram_sources/Figure_01_workflow.drawio` |
| 2 | `Figure_02_data_pipeline.png` | `src/generate_study_workflow_figures.py` | `figures/diagram_sources/Figure_02_data_pipeline.drawio` |
| 3 | `Figure_03_marker_frame.png` | Original anonymised illustrative frame embedded in the manuscript | — (photographic frame, not a diagram) |
| 4 | `Figure_04_leakage_controlled_ML_workflow.png` | `src/generate_model_figures.py` | `figures/diagram_sources/Figure_04_leakage_controlled_ML_workflow.drawio` |
| 5 | `Figure_05_repeated_nested_cv.png` | `src/generate_model_figures.py` | `figures/diagram_sources/Figure_05_repeated_nested_cv.drawio` |
| 6 | `Figure_06_pain_prevalence.png` | `src/generate_descriptive_figures.py`; deidentified counts CSV | — (data-driven plot) |
| 7 | `Figure_07_RULA_right.png` | `src/generate_descriptive_figures.py`; analytical workbook | — (data-driven plot) |
| 8 | `Figure_08_RULA_left.png` | `src/generate_descriptive_figures.py`; analytical workbook | — (data-driven plot) |
| 9 | `Figure_09_risk_class_distribution.png` | `src/generate_descriptive_figures.py`; analytical workbook | — (data-driven plot) |
| 10 | `Figure_10_model_balanced_accuracy.png` | `src/generate_model_figures.py`; primary result CSVs | — (data-driven plot) |
| 11 | `Figure_11_ablation_and_sensitivity.png` | `src/generate_model_figures.py`; primary/secondary result CSVs | — (data-driven plot) |
| 12 | `Figure_12_learning_curve.png` | `src/run_learning_curve.py`; learning-curve outputs | — (data-driven plot) |

Figures 1, 2, 4 and 5 are hand-authored workflow/design diagrams; their editable draw.io sources are under `figures/diagram_sources/` and open directly at [app.diagrams.net](https://app.diagrams.net). Figures 3 and 6–12 are either a photographic frame or generated directly from data and have no separate diagram source.

The legacy logistic-regression coefficient matrix was excluded from the main manuscript because it was produced outside nested validation and did not determine predictor selection. It is retained as `figures/supplementary/Figure_S1_legacy_logistic_coefficients.png` for audit purposes.
