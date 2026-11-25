"""End-to-end smoke test on SYNTHETIC data, in a temporary copy of the repo.

    python tests/smoke_test.py

Checks that every pipeline step runs and produces its outputs, that the
leakage audit passes, and that --drop-reversed-orders removes the reversed
80,995-unit order. It never touches your results/ folder.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXPECTED = [
    "cleaning_funnel.csv", "cleaning_summary.json", "eda_summary.json", "monthly_revenue.csv",
    "customer_purchasing_patterns.csv", "rfm_summary.json", "customer_rfm.csv", "kmeans_metrics.csv",
    "hierarchical_metrics.csv", "dbscan_grid.csv", "kmeans_stability.csv", "clustering_comparison.json",
    "k_selection.csv", "segment_profiles.csv", "customer_segments.csv", "metrics.csv",
    "classification_leakage_audit.json", "figures/segments_pca.png", "figures/dendrogram_ward.png",
    "figures/dbscan_kdistance.png",
]


def run(cmd, cwd):
    subprocess.run([sys.executable, *cmd], cwd=cwd, check=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(REPO, work, ignore=shutil.ignore_patterns(".git", "data", "results", "__pycache__"))
        for folder in ["data/raw", "data/processed", "results/figures"]:
            (work / folder).mkdir(parents=True, exist_ok=True)
        run(["tests/make_synthetic_workbook.py", "data/raw/online_retail_II.xlsx"], work)
        run(["src/run_all.py"], work)
        missing = [name for name in EXPECTED if not (work / "results" / name).exists()]
        assert not missing, f"missing outputs: {missing}"
        assert (work / "report" / "report.md").exists(), "report draft missing"
        audit = json.loads((work / "results/classification_leakage_audit.json").read_text())
        assert audit["check_all_feature_rows_before_window_end"] and audit["check_no_lifetime_rfm_in_features"]
        cleaning = json.loads((work / "results/cleaning_summary.json").read_text())
        assert "cleaning_retention_percent" in cleaning
        expected_retention = cleaning["clean_rows"] / cleaning["raw_rows"] * 100
        assert abs(cleaning["cleaning_retention_percent"] - expected_retention) < 1e-9
        cleaned_baseline = pd.read_csv(work / "data/processed/transactions_clean.csv")
        assert np.isfinite(cleaned_baseline[["Quantity", "Price"]].to_numpy()).all(), (
            "cleaned data contains non-finite Quantity or Price"
        )

        # Direct regression test: non-finite core numeric values must be removed.
        spec = importlib.util.spec_from_file_location(
            "clean_retail_data", work / "src/clean_retail_data.py"
        )
        cleaner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cleaner)
        bad_values = pd.DataFrame([
            ["700001", "20001", "VALID ITEM", 2, "2011-01-01 10:00", 3.5, 12001, "United Kingdom"],
            ["700002", "20002", "INFINITE QTY", np.inf, "2011-01-01 10:01", 3.5, 12002, "United Kingdom"],
            ["700003", "20003", "INFINITE PRICE", 2, "2011-01-01 10:02", np.inf, 12003, "United Kingdom"],
        ], columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
                    "Price", "Customer ID", "Country"])
        finite_cleaned, finite_funnel, _ = cleaner.clean_transactions(bad_values)
        assert len(finite_cleaned) == 1, "non-finite Quantity/Price values were not removed"
        invalid_stage = next(row for row in finite_funnel if row["rule"] == "invalid_core_fields")
        assert invalid_stage["rows_removed"] == 2, "invalid-core-fields funnel count is incorrect"

        metrics = pd.read_csv(work / "results/metrics.csv")
        assert {"model", "task", "split", "metric", "value", "notes"} == set(metrics.columns)
        assert metrics.value.between(0, 1).all(), "metric out of [0, 1]"
        baseline = metrics[(metrics.model == "Majority baseline") & (metrics.metric == "macro_f1")].value.max()
        best = metrics[(metrics.split == "test") & (metrics.metric == "macro_f1")].value.max()
        assert best > baseline, "no model beats the majority baseline"

        base_rows = json.loads((work / "results/cleaning_summary.json").read_text())["clean_rows"]
        run(["src/clean_retail_data.py", "--drop-reversed-orders"], work)
        new = json.loads((work / "results/cleaning_summary.json").read_text())
        assert new["reversed_order_lines_detected"] > 0, "synthetic reversed purchase was not detected"
        assert new["reversed_order_lines_removed"] == new["reversed_order_lines_detected"], (
            "detected reversed purchase lines were not all removed"
        )
        assert new["clean_rows"] == base_rows - new["reversed_order_lines_removed"], (
            "cleaning row count does not match the reversed-order removal count"
        )
        cleaned = pd.read_csv(work / "data/processed/transactions_clean.csv")
        assert not (cleaned["Quantity"] == 80995).any(), "reversed 80,995-unit order still present"
    print("\nSMOKE TEST PASSED (synthetic data; no real results were produced).")


if __name__ == "__main__":
    main()
