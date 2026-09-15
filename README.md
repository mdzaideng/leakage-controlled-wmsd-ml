<div align="center">

<h1> Leakage-Controlled WMSD-ML</h1>

<p><em>Leakage-Controlled Machine Learning for Work-Related Musculoskeletal Disorder Risk<br>
Classification Among Garment Sewing-Machine Operators</em></p>

<br>

[![DOI](https://zenodo.org/badge/1371403278.svg)](https://doi.org/10.5281/zenodo.22770247)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e)
![Nested CV](https://img.shields.io/badge/Nested_CV-10×_Repeated_5--Fold-6366f1)
![Status](https://img.shields.io/badge/Status-Under_Review-f59e0b)

<br>

**Version DOI** `10.5281/zenodo.22770248` · **All-versions DOI** `10.5281/zenodo.22770247`

</div>

---

> [!IMPORTANT]
> **Provisional outputs notice** — This repository accompanies a manuscript currently **under review** at *Results in Engineering*. All numerical results (tables, figures, metrics) in `outputs/` are provisional and may change following peer review. The code, notebooks, and leakage-control methodology are stable.

---

## Table of Contents

- [Overview](#-overview)
- [At a Glance](#-at-a-glance)
- [Key Finding](#-key-finding)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Run Order](#-run-order)
- [Leakage Controls](#️-leakage-controls)
- [RULA Correction Record](#-rula-correction-record)
- [Data](#-data)
- [Reproducibility Seed](#-reproducibility-seed)
- [License](#-license)
- [Citation](#-citation)
- [Authors](#-authors)

---

## Overview

This repository is the complete reproducibility package for a study that applies **leakage-controlled nested cross-validation** to classify WMSD risk among garment sewing-machine operators. Every step — from raw RULA postural scoring through hyperparameter tuning and model evaluation — is isolated to prevent data leakage, ensuring honest generalisation estimates on a 60-record occupational dataset.

---

## At a Glance

| Item | Detail |
|------|--------|
| **Participants** | 60 garment sewing-machine operators |
| **Labels** | RULA risk categories (time-weighted) |
| **CV scheme** | 10 × repeated 5-fold outer / 3-fold inner |
| **Primary metric** | Balanced accuracy |
| **Models compared** | LR, KNN, SVM, DT, RF, GB, XGB, MLP |
| **Leakage controls** | 4 explicit controls (see below) |
| **Seed** | `RANDOM_STATE = 42` throughout |

---

## Key Finding

> Random Forest with leakage-controlled nested cross-validation achieved the highest balanced accuracy, demonstrating that proper leakage prevention substantially lowers optimistic bias compared to conventional flat cross-validation.

*(Full results provisional pending peer review.)*

---

## Repository Structure

WMSD-leakage-controlled-reanalysis/
├── data/
│ └── WMSD_analysis_deidentified.xlsx # 60-record deidentified dataset
├── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_rula_scoring.ipynb
│ ├── 03_nested_cv_training.ipynb
│ ├── 04_results_analysis.ipynb
│ └── 05_visualisation.ipynb
├── outputs/
│ ├── figures/ # Publication figures (provisional)
│ └── tables/ # Metric tables (provisional)
├── diagrams/
│ └── *.drawio # Editable workflow diagram sources
├── CHECKSUMS.sha256 # SHA-256 over 76 tracked files
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt


---

## Installation

```bash
# Clone
git clone https://github.com/mdzaideng/leakage-controlled-wmsd-ml.git
cd leakage-controlled-wmsd-ml

# Create environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Run Order

Execute the notebooks **in sequence**; each saves intermediate artefacts consumed by the next.

01_data_preprocessing → 02_rula_scoring → 03_nested_cv_training
↓
05_visualisation ← 04_results_analysis


To reproduce all outputs end-to-end:

```bash
jupyter nbconvert --to notebook --execute notebooks/0*.ipynb --inplace
```

---

## Leakage Controls

Four explicit controls prevent information leakage between training and test folds:

| # | Control | Where Applied |
|---|---------|---------------|
| 1 | **Scaler fit inside inner fold only** | `03_nested_cv_training.ipynb` |
| 2 | **SMOTE applied inside inner fold only** | `03_nested_cv_training.ipynb` |
| 3 | **Feature selection inside inner fold only** | `03_nested_cv_training.ipynb` |
| 4 | **Hyperparameters selected on inner CV; outer fold is blind test** | `03_nested_cv_training.ipynb` |

---

## RULA Correction Record

Time-weighted RULA scores were recomputed from raw task-duration observations to correct two previously identified scoring errors. The correction methodology and before/after comparison are fully documented in `02_rula_scoring.ipynb`.

---

## Data

`data/WMSD_analysis_deidentified.xlsx` contains 60 records with all personally identifiable information removed. Reuse of this dataset is governed by the data-availability statement in the associated manuscript. Please cite the paper if you use the data.

---

## Reproducibility Seed

```python
RANDOM_STATE = 42   # applied to train/test splits, SMOTE, and all estimators
```

All stochastic operations use this seed. Running on the same hardware and Python version should reproduce results exactly; minor floating-point differences may arise across platforms.

---

## License

Code, notebooks, and diagram sources are released under the **MIT License** — see [`LICENSE`](LICENSE).

The deidentified dataset (`data/WMSD_analysis_deidentified.xlsx`) and derived result files are provided for reproducibility purposes alongside the associated manuscript; their reuse is governed by the data-availability statement in that manuscript.

---

## Citation

If you use this code or data, please cite:

```bibtex
@software{Hossain_Leakage-Controlled_WMSD-ML_Reproducibility_2026,
  author  = {Hossain, Md. Zaid and Arif, Borhan Ul},
  doi     = {10.5281/zenodo.22770248},
  license = {MIT},
  month   = sep,
  title   = {{Leakage-Controlled WMSD-ML: Reproducibility Package for Work-Related
              Musculoskeletal Disorder Risk Classification Among Garment
              Sewing-Machine Operators}},
  url     = {https://github.com/mdzaideng/leakage-controlled-wmsd-ml},
  version = {1.0.0},
  year    = {2026}
}
```

---

## Authors

<table>
<tr>
<td width="50%" valign="top">

**Md. Zaid Hossain**<br>
Department of Industrial Engineering and Management<br>
Khulna University of Engineering and Technology<br>
Khulna-9203, Bangladesh<br><br>
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--3301--3609-a6ce39?style=flat-square&logo=orcid&logoColor=white)](https://orcid.org/0009-0003-3301-3609)

</td>
<td width="50%" valign="top">

**Borhan Ul Arif**<br>
Department of Computer Science and Engineering<br>
Southeast University<br>
Dhaka, Bangladesh<br><br><br>
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--0536--8305-a6ce39?style=flat-square&logo=orcid&logoColor=white)](https://orcid.org/0009-0007-0536-8305)

</td>
</tr>
</table>

---

<div align="center">
<sub>Released under the MIT License · Zenodo DOI <a href="https://doi.org/10.5281/zenodo.22770247">10.5281/zenodo.22770247</a></sub>
</div>
