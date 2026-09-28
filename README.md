# Unified static dielectric tensor (Unified ε) for vdW NbOX₂ films — data & code

Companion deposit for the manuscript "Unified static ε from MACEField polarizability
on NbOX2 film structures: Tr(ε) DFT hybrid with α-anisotropy, ALIGNN and GNNOpt
calibration to HSE06+LOPTICS, and the memristor/LED device context".

## Contents
- `scripts/` — calibration and visualization code (Python 3.11+):
  - `ml_eg_jarvis_scissor_calibrate.py` — Phase A band-gap scissor calibration on paired JARVIS-DFT labels;
  - `ml_eps_spectral_scissor_calibrate.py` — Phase A/C spectral calibration of Tr ε(ω) (schemes I–III, Godby scissors);
  - `ml_dgl_bootstrap.py` — DGL import bootstrap for Windows / torch 2.7 (frozen ALIGNN inference);
  - `ml_gnn_pipeline_viz.py` — pedagogical visualization of the INPUT | graph | OUTPUT pipeline.
- `data/` — reference and derived datasets:
  - `NbOI2_HSE06_dielectric_tensor_energy.csv` — authors' VASP HSE06+LOPTICS reference spectra for bulk NbOI2 (εxx, εyy, εzz vs energy);
  - `dielectric_compare` outputs (CSV/JSON) — side-by-side Mortazavi HSE06+RPA vs MACE Unified ε exports and ALIGNN Eg results.
- `figures/` — manuscript figures (English labels).

## Provenance
- Reference DFT: authors' own VASP HSE06+LOPTICS run on bulk nboi2.xyz (16 atoms, C2), ~49,850 core·h (unpublished; this deposit).
- Literature anchors: Mortazavi et al., Nanotechnology 33, 275701 (2022), doi:10.1088/1361-6528/ac622f;
  Li et al., ACS Nano (2025), doi:10.1021/acsnano.5c07236.
- ML models (frozen inference, no retraining): MACEField (arXiv:2508.17870), ALIGNN and GNNOpt
  (trained on JARVIS-DFT; Choudhary et al.).

## Key metrics (reference iodine film, Unified mode)
α ≈ (4.73, 6.40, 8.15) Å³ → Unified ε ≈ (4.56, 6.17, 7.86), Tr(ε) = 18.59;
cos(f) ≈ 0.939 vs Mortazavi HSE06+RPA and ≈ 0.945 vs the authors' HSE06+LOPTICS CSV.
