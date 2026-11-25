# Reproducibility

## Environment
- Python version: 3.11 or 3.12. The code was tested on Python 3.12.3 with the exact versions in `requirements.txt` (pandas 2.2.3, NumPy 2.1.3, scikit-learn 1.5.2, SciPy 1.14.1, Matplotlib 3.9.2, openpyxl 3.1.5) and also with newer releases.
- OS and hardware used for the final real-data run: TODO (record before submission).
- Install dependencies: `python -m pip install -r requirements.txt`

## Dataset
1. Download UCI Online Retail II from https://archive.ics.uci.edu/dataset/502/online+retail+ii using a browser.
2. Extract the workbook to `data/raw/online_retail_II.xlsx`.
3. Do not disable TLS/SSL verification to work around download problems.

## Run sequence
Run from the repository root:

1. `python src/clean_retail_data.py`
2. `python src/run_eda.py`
3. `python src/build_rfm.py`
4. `python src/clustering.py`
5. `python src/clustering_advanced.py`
6. `python src/segments.py`
7. `python src/classification.py`
8. `python src/make_report.py`

Or all at once: `python src/run_all.py` (use `--from-step N` to resume). Notebooks in `notebooks/` (`01-eda`, `02-preprocessing`, `03-models`, `04-evaluation`) only read the files in `results/`, so run the scripts first. Expected runtime on the full data: TODO (record from the final run; classification is the slowest step because of cross-validated grid searches).

`python tests/smoke_test.py` runs the whole pipeline on a small SYNTHETIC workbook in a temporary folder to check the code; it never writes to `results/` and its numbers must not be reported.

GitHub Actions runs the same synthetic smoke test on pushes and pull requests targeting `main` (`.github/workflows/smoke-test.yml`). This CI check validates the synthetic pipeline only; it does not rerun or validate the committed real-data results.

## Randomness
- All random seeds are `42`: K-Means (`common.RANDOM_STATE`, `n_init=20`), stability subsamples (seeded generator; each bootstrap model uses seed 43–62, `n_init=10`), the silhouette sample in `clustering.py`, PCA, the train/test split and cross-validation folds in `classification.py`, and the Random Forest/Decision Tree/Logistic Regression models.
- DBSCAN and hierarchical clustering are deterministic.
- Results can still vary slightly across scikit-learn/NumPy versions and CPU architectures.

## Reproducibility limitations
- Cancellation invoices are excluded rather than matched/netted against original invoices, so Monetary is gross sales and a few reversed orders (for example the 80,995-unit paper-craft order) can inflate a customer's value. `--drop-reversed-orders` is provided as a sensitivity check but is off by default.
- Extreme positive quantities are not automatically removed or capped; outliers are only flagged in EDA.
- Customer-level files (`customer_rfm.csv`, `rfm_scaled.csv`, `customer_segments.csv`, `kmeans_cluster_*.csv`, `customer_purchasing_patterns.csv`) and the processed transactions are generated locally and not committed.
- Segment names are heuristic labels and the final k comes from a documented rule that can be overridden with `--k`; both should be reviewed by the team.
- The raw workbook is not committed; the numbers depend on the UCI file being the unmodified Online Retail II workbook.
