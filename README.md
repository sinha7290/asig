# ASIG

**Autonomy Signature (ASIG): companion code for our PNAS Nexus 2024 paper on growth signaling autonomy in circulating tumor cells and metastatic seeding in breast cancer.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Publication](https://img.shields.io/badge/Published-PNAS%20Nexus%202024-blue)](https://doi.org/10.1093/pnasnexus/pgae014)
[![PMID](https://img.shields.io/badge/PMID-38312224-lightgrey)](https://pubmed.ncbi.nlm.nih.gov/38312224/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)

## Overview

This repository contains the code and derived data underlying our study of **growth signaling autonomy**, the first classical hallmark of cancer, in the context of blood-borne metastatic dissemination of human breast cancer.

Using breast cancer cell lines that are either endowed with, or CRISPR impaired for, a Golgi-localized two GTPase switch circuit (centered on GIV / Girdin / CCDC88A), we derive an **Autonomy Signature (ASIG)** from paired transcriptomic and proteomic profiles. We then show that ASIG:

* Reflects a distinct cellular state induced under growth factor restriction, tightly coupled to self-sustained EGFR / ErbB signaling
* Is enriched for programs of stemness, proliferation, and epithelial mesenchymal plasticity (EMP)
* Is uniquely induced in **circulating tumor cells (CTCs)**, the phase of the tumor life cycle most deprived of exogenous EGF
* Tracks therapeutic response and prognosticates outcome in breast cancer patient cohorts

The workflow uses the **Boolean implication framework** (StepMiner, Hegemon, BoNE) to build the signature and score external cohorts, followed by survival analyses in CTC transcriptomic datasets.

## Repository contents

| File | Purpose |
| :--- | :--- |
| `ASIG composite score.ipynb` | Derive and compute the ASIG composite score from transcriptomic data |
| `Asig_bubble_plots.ipynb` | Generate publication bubble plots of ASIG behavior across cohorts and conditions |
| `KM_CTC.ipynb` | Kaplan Meier survival analyses stratified by ASIG in CTC transcriptomic datasets |
| `HegemonUtil.py` | Utility module for the Hegemon Boolean analysis framework |
| `StepMiner.py` | StepMiner implementation for identifying step patterns in gene expression |
| `bone.py` | BoNE (Boolean Network Explorer) helper functions used for signature scoring |

## Reproducing the analysis

1. Clone the repository and install dependencies (Python 3.9+, standard scientific stack, plus `lifelines` for survival analyses).
2. Obtain the transcriptomic datasets used in the paper from GEO. Accessions and preprocessing steps are documented within each notebook.
3. Run the notebooks in the following order:
   1. `ASIG composite score.ipynb` to build the signature and compute composite scores.
   2. `KM_CTC.ipynb` to reproduce the survival analyses in CTC cohorts.
   3. `Asig_bubble_plots.ipynb` to reproduce the summary bubble plots.

## Key concept: the Autonomy Signature

Growth signaling autonomy is the ability of a tumor cell to secrete and sense growth factors in a self-sustained autocrine and paracrine loop. This state confers survival advantages under harsh conditions such as circulation, where systemic EGF is largely bound to platelets and unavailable to disseminated cells. ASIG captures this state at the transcriptomic level and enables its detection in independent patient cohorts.

## Publication

> Sinha S, Farfel A, Luker KE, Parker BA, Yeung KT, Luker GD, Ghosh P.
> **Growth signaling autonomy in circulating tumor cells aids metastatic seeding.**
> *PNAS Nexus*, 2024 Jan 25; 3(2): pgae014.
> doi: [10.1093/pnasnexus/pgae014](https://doi.org/10.1093/pnasnexus/pgae014)
> PMID: 38312224 · PMCID: PMC10833458

If you use this code, the ASIG signature, or derived analyses in your own work, please cite the manuscript.

## Contact

**Saptarshi Sinha, Ph.D.**
Assistant Project Scientist, Department of Cellular and Molecular Medicine
Director, PreCSN Center
University of California San Diego
Email: sasinha@health.ucsd.edu

## License

Released under the MIT License. See `LICENSE`.
