"""Transparent first-pass cleaning for UCI Online Retail II.

Usage:
    python src/clean_retail_data.py
    python src/clean_retail_data.py --input data/raw/online_retail_II.xlsx

Review the rules with the team before treating this baseline as final.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

REQUIRED_COLUMNS = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "Customer ID", "Country",
]
# Explicit service, adjustment and non-merchandise codes excluded in Phase 1.
NON_PRODUCT_CODES = {
    "POST", "M", "DOT", "BANK CHARGES", "AMAZONFEE", "ADJUST",
    "CRUK", "D", "C2", "S", "PADS", "B", "TEST001",
}


def read_workbook(path: Path) -> pd.DataFrame:
    """Read both source sheets and validate the expected schema."""
    if not path.exists():
        raise FileNotFoundError(
            f"Workbook not found: {path}\n"
            "Download UCI Online Retail II in a browser, extract the ZIP, "
            "and place online_retail_II.xlsx in data/raw/."
        )
    book = pd.ExcelFile(path, engine="openpyxl")
    if len(book.sheet_names) < 2:
        raise ValueError(f"Expected two workbook sheets; found {book.sheet_names}")
    frames = []
    for sheet in book.sheet_names[:2]:
        frame = pd.read_excel(book, sheet_name=sheet, engine="openpyxl")
        frame.columns = [str(column).strip() for column in frame.columns]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}; found {list(data.columns)}")
    return data


def record_step(funnel: list[dict], name: str, before: int, after: int, decision: str) -> None:
    funnel.append({
        "rule": name,
        "rows_before": int(before),
        "rows_after": int(after),
        "rows_removed": int(before - after),
        "decision": decision,
    })


def keep_rows(
    data: pd.DataFrame, funnel: list[dict], name: str,
    mask: pd.Series, decision: str,
) -> pd.DataFrame:
    before = len(data)
    result = data.loc[mask].copy()
    record_step(funnel, name, before, len(result), decision)
    return result


def find_reversed_orders(data: pd.DataFrame) -> pd.Index:
    """Index labels of purchase lines later reversed by a cancellation line.

    A cancellation line (Invoice starts with C, Quantity < 0) is matched to the
    latest earlier purchase line with the same Customer ID, StockCode and
    quantity. Used only when --drop-reversed-orders is set.
    """
    is_cancel = data["Invoice"].str.startswith("C", na=False)
    cols = ["Customer ID", "StockCode", "Quantity", "InvoiceDate"]
    valid_match_keys = (
        data["Customer ID"].notna()
        & data["StockCode"].notna()
        & data["InvoiceDate"].notna()
    )
    purchases = data.loc[
        valid_match_keys & ~is_cancel & (data["Quantity"] > 0), cols
    ].copy()
    cancels = data.loc[
        valid_match_keys & is_cancel & (data["Quantity"] < 0), cols
    ].copy()
    if purchases.empty or cancels.empty:
        return pd.Index([])
    cancels["Quantity"] = -cancels["Quantity"]
    for frame in (purchases, cancels):
        frame["StockCode"] = frame["StockCode"].astype(object)
    purchases["purchase_index"] = purchases.index
    purchases = purchases.sort_values("InvoiceDate")
    cancels = cancels.sort_values("InvoiceDate")
    matched = pd.merge_asof(
        cancels, purchases, on="InvoiceDate",
        by=["Customer ID", "StockCode", "Quantity"], direction="backward",
    )
    return pd.Index(matched["purchase_index"].dropna().astype(int).unique())


def clean_transactions(
    raw: pd.DataFrame, drop_reversed: bool = False
) -> tuple[pd.DataFrame, list[dict], dict]:
    data = raw.copy()
    funnel: list[dict] = []
    diagnostics = {
        "raw_rows": int(len(data)),
        "raw_missing_customer_id_rows": int(data["Customer ID"].isna().sum()),
        "raw_missing_customer_id_percent": float(data["Customer ID"].isna().mean() * 100),
        "raw_cancellation_rows": int(
            data["Invoice"].astype("string").str.startswith("C", na=False).sum()
        ),
    }

    for column in ["Invoice", "StockCode", "Description", "Country"]:
        data[column] = data[column].astype("string").str.strip()
    data["Customer ID"] = pd.to_numeric(data["Customer ID"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")

    diagnostics["raw_duplicate_rows"] = int(data.duplicated().sum())  # after type coercion, as removed below
    data = keep_rows(
        data, funnel, "exact_duplicates", ~data.duplicated(keep="first"),
        "Keep the first copy of exact duplicate rows.",
    )
    data = keep_rows(
        data, funnel, "missing_customer_id", data["Customer ID"].notna(),
        "Remove rows without Customer ID for customer-level segmentation; report lost share.",
    )
    cancellation = data["Invoice"].str.startswith("C", na=False)
    reversed_index = find_reversed_orders(data) if drop_reversed else pd.Index([])
    diagnostics["reversed_order_lines_detected"] = int(len(reversed_index))
    diagnostics["reversed_order_lines_removed"] = 0
    data = keep_rows(
        data, funnel, "cancellation_invoices", ~cancellation,
        "Baseline excludes cancellation invoices rather than netting them to original invoices.",
    )
    if drop_reversed:
        before = len(data)
        keep_mask = ~data.index.isin(reversed_index)
        data = keep_rows(
            data, funnel, "reversed_purchase_lines", keep_mask,
            "Sensitivity option: remove earlier purchase lines matched to later cancellation lines "
            "for the same customer, stock code and quantity.",
        )
        diagnostics["reversed_order_lines_removed"] = int(before - len(data))
    # Reject missing values and +/- infinity; > 0 alone would allow
    # infinite quantities or prices into revenue and RFM calculations.
    valid_core = (
        data["InvoiceDate"].notna()
        & data["Invoice"].notna()
        & data["StockCode"].notna()
        & np.isfinite(data["Quantity"])
        & np.isfinite(data["Price"])
    )
    data = keep_rows(
        data, funnel, "invalid_core_fields", valid_core,
        "Remove rows with missing/invalid invoice, stock code, date, quantity or price.",
    )
    data = keep_rows(
        data, funnel, "non_product_stock_codes",
        ~data["StockCode"].str.upper().isin(NON_PRODUCT_CODES),
        "Exclude known service/adjustment codes; review unique codes during EDA.",
    )
    data = keep_rows(
        data, funnel, "missing_descriptions", data["Description"].notna(),
        "Remove rows without a product description.",
    )
    data = keep_rows(
        data, funnel, "non_positive_quantity", data["Quantity"] > 0,
        "Keep positive quantities; cancellation invoices were already excluded.",
    )
    data = keep_rows(
        data, funnel, "non_positive_price", data["Price"] > 0,
        "Keep positive unit prices.",
    )

    diagnostics.update({
        "clean_rows": int(len(data)),
        "cleaning_retention_percent": (
            float(len(data) / len(raw) * 100) if len(raw) else 0.0
        ),
        "cleaning_removed_rows": int(len(raw) - len(data)),
        "clean_customers": int(data["Customer ID"].nunique()),
        "clean_min_invoice_date": data["InvoiceDate"].min().isoformat() if len(data) else None,
        "clean_max_invoice_date": data["InvoiceDate"].max().isoformat() if len(data) else None,
        "clean_total_revenue_gbp": float((data["Quantity"] * data["Price"]).sum()),
        "clean_unique_stock_codes": int(data["StockCode"].nunique()),
        "limitations": [
            "Rules are a baseline and must be reviewed against the actual data.",
            "Cancellation invoices are excluded rather than netted against original invoices.",
            "The non-product code list may need expansion after reviewing unique StockCode values.",
            "Extreme positive quantities are not automatically removed or capped; inspect outliers.",
            "Customer ID is an identifier, not a continuous model feature.",
        ],
    })
    return data, funnel, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--drop-reversed-orders", action="store_true",
        help="Sensitivity option: also drop purchase lines later reversed by a "
             "cancellation (off by default so the documented baseline is unchanged).",
    )
    args = parser.parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading workbook: {args.input}")
    raw = read_workbook(args.input)
    print(f"Combined raw shape: {raw.shape}")
    cleaned, funnel, diagnostics = clean_transactions(raw, args.drop_reversed_orders)

    cleaned.to_csv(PROCESSED_DIR / "transactions_clean.csv", index=False)
    pd.DataFrame(funnel).to_csv(RESULTS_DIR / "cleaning_funnel.csv", index=False)
    with (RESULTS_DIR / "cleaning_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, ensure_ascii=False)

    print("\nCleaning funnel:")
    print(pd.DataFrame(funnel).to_string(index=False))
    print("\nDiagnostics:")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print("\nWrote cleaned transactions and cleaning diagnostics.")


if __name__ == "__main__":
    main()
