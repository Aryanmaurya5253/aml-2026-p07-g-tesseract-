"""Phase 3: build customer-level Recency, Frequency and Monetary features."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FEATURES = ["Recency", "Frequency", "Monetary"]


def main() -> None:
    input_path = PROCESSED / "transactions_clean.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {input_path}")

    data = pd.read_csv(input_path, parse_dates=["InvoiceDate"])
    data["Revenue"] = data["Quantity"] * data["Price"]
    reference_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = data.groupby("Customer ID").agg(
        last_purchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
    ).reset_index()
    rfm["Recency"] = (reference_date - rfm["last_purchase"]).dt.days
    rfm = rfm[["Customer ID", *FEATURES]]
    values = rfm[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("RFM features must be finite and positive before log scaling.")

    log_values = np.log1p(rfm[FEATURES])
    scaled_values = StandardScaler().fit_transform(log_values)
    scaled = rfm[["Customer ID"]].copy()
    scaled[FEATURES] = scaled_values

    RESULTS.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(RESULTS / "customer_rfm.csv", index=False)
    scaled.to_csv(RESULTS / "rfm_scaled.csv", index=False)

    raw_stats = {}
    log_stats = {}
    for feature in FEATURES:
        raw = rfm[feature]
        logged = log_values[feature]
        raw_stats[feature] = {
            "mean": float(raw.mean()),
            "median": float(raw.median()),
            "max": float(raw.max()),
            "skewness": float(raw.skew()),
        }
        log_stats[feature] = {
            "mean": float(logged.mean()),
            "median": float(logged.median()),
            "max": float(logged.max()),
            "skewness": float(logged.skew()),
        }
    pd.DataFrame(raw_stats).T.reset_index(names="feature").to_csv(
        RESULTS / "rfm_feature_statistics.csv", index=False
    )
    pd.DataFrame(log_stats).T.reset_index(names="feature").to_csv(
        RESULTS / "rfm_log_feature_statistics.csv", index=False
    )
    summary = {
        "customer_count": int(len(rfm)),
        "reference_date": reference_date.isoformat(),
        "features": FEATURES,
        "definitions": {
            "Recency": "Days from the fixed reference date to the customer's last purchase.",
            "Frequency": "Number of distinct invoices.",
            "Monetary": "Gross retained revenue in GBP.",
        },
        "raw_feature_statistics": raw_stats,
        "log_feature_skewness": {key: value["skewness"] for key, value in log_stats.items()},
        "scaling": "StandardScaler fitted to log1p(RFM) features.",
    }
    (RESULTS / "rfm_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Save raw and log1p distributions for all three RFM features.
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for column, feature in enumerate(FEATURES):
        axes[0, column].hist(rfm[feature], bins=50)
        axes[0, column].set_title(f"{feature} (raw)")
        axes[0, column].set_xlabel(feature)
        axes[1, column].hist(log_values[feature], bins=50)
        axes[1, column].set_title(f"log1p({feature})")
        axes[1, column].set_xlabel(f"log1p({feature})")
    fig.suptitle("Customer RFM distributions before and after log transformation")
    fig.tight_layout()
    fig.savefig(FIGURES / "rfm_distributions.png", dpi=160)
    plt.close(fig)

    print(f"Customers: {len(rfm):,}; reference date: {reference_date.date()}")
    print("RFM tables and summary saved under results/.")


if __name__ == "__main__":
    main()
