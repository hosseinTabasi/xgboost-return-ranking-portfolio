# Notebooks

Figures and tables are produced by `python -m src.main` (Agg backend) and written to `results/figures/` and `results/tables/`. Open those files rather than re-running the download inside a notebook.

Primary artefacts:

- `results/figures/equity_curves.png`
- `results/figures/feature_importance.png`
- `results/figures/shap_summary.png` (if SHAP ran)
- `results/tables/performance.csv`
- `results/tables/subperiods.csv`

`01_results.ipynb` only loads those CSVs.
