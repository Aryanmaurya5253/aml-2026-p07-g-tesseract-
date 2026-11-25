"""Phase 6/7: predict a customer's segment from EARLY purchase behaviour only.

Run from the repository root after src/segments.py:
    python src/classification.py
    python src/classification.py --window-days 60 --label-source future

Leakage design
--------------
* Features come only from a customer's first WINDOW_DAYS days after their first
  purchase (InvoiceDate < first purchase date + window). No lifetime
  Recency/Frequency/Monetary and no calendar/acquisition-month feature is used.
* Only customers whose full early window lies inside the data are eligible.
* All scaling/log transforms sit inside sklearn Pipelines fitted on the training
  split only; hyper-parameters are tuned by cross-validation on training data.
* Two label sources are evaluated:
  - future   (strict, leakage-free): K-Means segments built ONLY from activity in
             the HORIZON_DAYS after the early window (frequency, spend, recency
             within that fixed-length horizon), so labels and features share no
             transactions and acquisition date cannot leak into the labels.
             Customers need a complete early window AND a complete horizon.
             This is the headline estimate.
  - lifetime (project question): the Phase 5 lifetime segments. These labels are
             computed from all transactions, including the early window, so this
             number is optimistic and is reported for comparison only.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import FEATURES, FIGURES, PROCESSED, RANDOM_STATE, RESULTS, fit_kmeans

LOG_COLS = [
    "early_invoices", "early_lines", "early_units", "early_spend",
    "early_avg_invoice_value", "early_max_invoice_value", "first_invoice_value",
    "first_invoice_lines", "early_distinct_products", "early_active_days",
    "early_units_per_line", "early_avg_unit_price", "early_days_span",
]
PASS_COLS = ["is_uk", "first_weekday", "first_hour"]
EARLY_FEATURES = LOG_COLS + PASS_COLS
MIN_CLASS_COUNT = 20  # classes smaller than this cannot be evaluated reliably
METRIC_COLUMNS = ["model", "task", "split", "metric", "value", "notes"]


def load_transactions():
    path = PROCESSED / "transactions_clean.csv"
    if not path.exists():
        raise FileNotFoundError("Run python src/clean_retail_data.py first.")
    tx = pd.read_csv(path, parse_dates=["InvoiceDate"])
    tx["Revenue"] = tx["Quantity"] * tx["Price"]
    return tx.sort_values(["Customer ID", "InvoiceDate"]).reset_index(drop=True)


def split_windows(tx, window_days, horizon_days):
    """Return (early, post, eligible_ids, reference_date, tx_with_window_end)."""
    reference = tx["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    first = tx.groupby("Customer ID")["InvoiceDate"].transform("min")
    tx = tx.assign(window_end=first.dt.normalize() + pd.Timedelta(days=window_days))
    eligible = (tx.groupby("Customer ID")["window_end"].first()
                + pd.Timedelta(days=horizon_days)) <= reference
    early = tx[tx["InvoiceDate"] < tx["window_end"]]
    post = tx[tx["InvoiceDate"] >= tx["window_end"]]
    return early, post, eligible[eligible].index, reference, tx


def early_features(early):
    invoices = early.groupby(["Customer ID", "Invoice"]).agg(
        invoice_value=("Revenue", "sum"), invoice_lines=("Revenue", "size"),
        invoice_date=("InvoiceDate", "min"), country=("Country", "first"),
    ).reset_index().sort_values(["Customer ID", "invoice_date"])
    first_invoice = invoices.groupby("Customer ID").first()
    feats = early.groupby("Customer ID").agg(
        early_invoices=("Invoice", "nunique"), early_lines=("Invoice", "size"),
        early_units=("Quantity", "sum"), early_spend=("Revenue", "sum"),
        early_distinct_products=("StockCode", "nunique"), early_avg_unit_price=("Price", "mean"),
        first_ts=("InvoiceDate", "min"), last_ts=("InvoiceDate", "max"),
    )
    feats["early_active_days"] = early.assign(day=early["InvoiceDate"].dt.normalize()).groupby(
        "Customer ID")["day"].nunique()
    feats["early_days_span"] = (feats["last_ts"] - feats["first_ts"]).dt.days
    feats["early_avg_invoice_value"] = feats["early_spend"] / feats["early_invoices"]
    feats["early_max_invoice_value"] = invoices.groupby("Customer ID")["invoice_value"].max()
    feats["early_units_per_line"] = feats["early_units"] / feats["early_lines"]
    feats["first_invoice_value"] = first_invoice["invoice_value"]
    feats["first_invoice_lines"] = first_invoice["invoice_lines"]
    feats["is_uk"] = (first_invoice["country"] == "United Kingdom").astype(int)
    feats["first_weekday"] = feats["first_ts"].dt.weekday
    feats["first_hour"] = feats["first_ts"].dt.hour
    return feats[EARLY_FEATURES]


def future_labels(post, window_end, ids, k, horizon_days):
    """K-Means segments from behaviour in a fixed horizon after the early window."""
    horizon_end = window_end.reindex(ids) + pd.Timedelta(days=horizon_days)
    sub = post[post["Customer ID"].isin(ids)].copy()
    sub["horizon_end"] = sub["Customer ID"].map(horizon_end)
    sub = sub[sub["InvoiceDate"] < sub["horizon_end"]]
    grouped = sub.groupby("Customer ID")
    frame = pd.DataFrame(index=ids)
    frame["future_frequency"] = grouped["Invoice"].nunique().reindex(ids).fillna(0)
    frame["future_monetary"] = grouped["Revenue"].sum().reindex(ids).fillna(0)
    last = grouped["InvoiceDate"].max().reindex(ids)
    since_last = (horizon_end - last.dt.normalize()).dt.days
    frame["future_recency"] = since_last.fillna(horizon_days).clip(lower=0, upper=horizon_days)
    logged = np.log1p(frame[["future_recency", "future_frequency", "future_monetary"]])
    labels = fit_kmeans(StandardScaler().fit_transform(logged), k).labels_
    order = (frame.assign(c=labels).groupby("c")["future_monetary"].median()
             .sort_values(ascending=False).index.tolist())
    rank = {c: i for i, c in enumerate(order)}
    names = pd.Series([f"Future-{rank[c]} (value rank {rank[c] + 1} of {k})" for c in labels], index=ids)
    profile = frame.assign(label=names).groupby("label").agg(
        customers=("future_monetary", "size"),
        future_recency_median=("future_recency", "median"),
        future_frequency_median=("future_frequency", "median"),
        future_monetary_median=("future_monetary", "median"),
        share_with_no_future_purchase=("future_frequency", lambda s: float((s == 0).mean())),
    ).reset_index()
    return names, profile


def make_preprocessor():
    log_pipe = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([("log", log_pipe, LOG_COLS), ("pass", StandardScaler(), PASS_COLS)])


def model_specs(seed):
    def pipe(clf):
        return Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    return {
        "Majority baseline": (pipe(DummyClassifier(strategy="most_frequent")), {}),
        "Logistic Regression": (
            pipe(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
            {"clf__C": [0.1, 1.0, 10.0]}),
        "Decision Tree": (
            pipe(DecisionTreeClassifier(class_weight="balanced", random_state=seed)),
            {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 20]}),
        "Random Forest": (
            pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                        n_jobs=1, random_state=seed)),
            {"clf__max_depth": [8, 16], "clf__min_samples_leaf": [1, 3, 5]}),
    }


def evaluate(task, X, y, seed, test_size):
    classes = sorted(y.unique())
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify)
    n_splits = int(max(2, min(5, y_tr.value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows, fitted, cv_scores = [], {}, {}
    for name, (estimator, grid) in model_specs(seed).items():
        if grid:
            # Keep CV serial to avoid spawning many memory-heavy workers on student laptops.
            search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, n_jobs=1)
            search.fit(X_tr, y_tr)
            best, best_cv = search.best_estimator_, float(search.best_score_)
            cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
            note = f"best params {search.best_params_}; {n_splits}-fold CV on train"
        else:
            best = estimator.fit(X_tr, y_tr)
            scores = cross_val_score(estimator, X_tr, y_tr, scoring="f1_macro", cv=cv)
            best_cv, cv_std = float(scores.mean()), float(scores.std())
            note = f"no tuning; {n_splits}-fold CV on train"
        fitted[name], cv_scores[name] = best, best_cv
        pred = best.predict(X_te)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, pred, labels=classes, zero_division=0)
        add = lambda split, metric, value, n="": rows.append(
            {"model": name, "task": task, "split": split, "metric": metric,
             "value": float(value), "notes": n})
        add("cv_train", "macro_f1_mean", best_cv, note)
        add("cv_train", "macro_f1_std", cv_std, note)
        add("test", "accuracy", accuracy_score(y_te, pred), f"n_test={len(y_te)}")
        add("test", "macro_f1", f1_score(y_te, pred, average="macro", zero_division=0))
        add("test", "weighted_f1", f1_score(y_te, pred, average="weighted", zero_division=0))
        add("test", "balanced_accuracy", balanced_accuracy_score(y_te, pred))
        for cls, p_, r_, f_ in zip(classes, prec, rec, f1):
            add("test", f"recall[{cls}]", r_, f"precision={p_:.4f}; f1={f_:.4f}")
    return classes, X_te, y_te, fitted, cv_scores, rows, (len(X_tr), len(X_te))


def plot_confusion(task, model_name, model, X_te, y_te, classes):
    pred = model.predict(X_te)
    cm = confusion_matrix(y_te, pred, labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        RESULTS / f"confusion_matrix_{task}.csv")
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    size = max(6, 1.1 * len(classes) + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > .5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix: {model_name}\n({task}; cells = count and row %)")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    fig.savefig(FIGURES / f"confusion_matrix_{task}.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window-days", type=int, default=30, help="early-behaviour window after first purchase")
    parser.add_argument("--horizon-days", type=int, default=90, help="future horizon used for the 'future' labels")
    parser.add_argument("--label-source", choices=["lifetime", "future", "both"], default="both")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    seg_path = RESULTS / "customer_segments.csv"
    seg_summary_path = RESULTS / "segment_summary.json"
    if not seg_path.exists() or not seg_summary_path.exists():
        raise FileNotFoundError("Run python src/segments.py first.")
    segments = pd.read_csv(seg_path).set_index("Customer ID")
    final_k = int(json.loads(seg_summary_path.read_text())["final_k"])
    FIGURES.mkdir(parents=True, exist_ok=True)

    tx = load_transactions()
    early, post, eligible_ids, reference, tx_w = split_windows(tx, args.window_days, args.horizon_days)
    early = early[early["Customer ID"].isin(eligible_ids)]
    feats = early_features(early)
    window_end = tx_w.groupby("Customer ID")["window_end"].first()

    # Leakage audit: every early row must precede its customer's window end, and
    # no lifetime RFM column may be a feature.
    audit_ok = bool((early["InvoiceDate"] < early["window_end"]).all()
                    and (post["InvoiceDate"] >= post["window_end"]).all())
    audit = {
        "window_days": args.window_days, "horizon_days": args.horizon_days,
        "eligible_customers": int(len(feats)),
        "customers_excluded_incomplete_window_or_horizon": int(tx["Customer ID"].nunique() - len(feats)),
        "features": EARLY_FEATURES,
        "excluded_by_design": ["Recency", "Frequency", "Monetary (lifetime)", "acquisition month/year",
                               "any transaction on/after the window end"],
        "check_all_feature_rows_before_window_end": audit_ok,
        "check_no_lifetime_rfm_in_features": not set(FEATURES) & set(EARLY_FEATURES),
        "preprocessing": "log1p + StandardScaler inside Pipelines fitted on the training split only",
        "reference_date": str(reference.date()),
    }
    assert audit_ok and audit["check_no_lifetime_rfm_in_features"], "Leakage audit failed"

    tasks = []
    if args.label_source in ("lifetime", "both"):
        y = segments["SegmentName"].reindex(feats.index).dropna()
        tasks.append(("segment_lifetime", y, None))
    if args.label_source in ("future", "both"):
        y, profile = future_labels(post, window_end, feats.index, final_k, args.horizon_days)
        profile.to_csv(RESULTS / "classification_future_label_profiles.csv", index=False)
        tasks.append(("segment_future", y, profile))

    all_rows, summary = [], {"audit": audit, "tasks": {}}
    for task, y, _ in tasks:
        counts = y.value_counts()
        dropped = {str(c): int(n) for c, n in counts[counts < MIN_CLASS_COUNT].items()}
        if dropped:
            print(f"Note: dropping classes with < {MIN_CLASS_COUNT} eligible customers: {dropped}")
            y = y[~y.isin(list(dropped))]
        X = feats.loc[y.index]
        print(f"\n=== {task}: {len(y):,} customers; classes: {y.value_counts().to_dict()}")
        classes, X_te, y_te, fitted, cv_scores, rows, sizes = evaluate(
            task, X, y, args.seed, args.test_size)
        all_rows.extend(rows)
        best_name = max((m for m in cv_scores if m != "Majority baseline"), key=cv_scores.get)
        plot_confusion(task, best_name, fitted[best_name], X_te, y_te, classes)
        rf = fitted["Random Forest"].named_steps["clf"]
        pd.DataFrame({"feature": LOG_COLS + PASS_COLS, "importance": rf.feature_importances_}) \
            .sort_values("importance", ascending=False) \
            .to_csv(RESULTS / f"feature_importance_{task}.csv", index=False)
        test = {(r["model"], r["metric"]): r["value"] for r in rows if r["split"] == "test"}
        summary["tasks"][task] = {
            "customers": int(len(y)), "train_size": sizes[0], "test_size": sizes[1],
            "class_counts": {k: int(v) for k, v in y.value_counts().items()},
            "dropped_small_classes": dropped,
            "best_model_by_cv_macro_f1": best_name,
            "test_macro_f1": {m: test[(m, "macro_f1")] for m in fitted},
            "test_accuracy": {m: test[(m, "accuracy")] for m in fitted},
        }
        for m in fitted:
            print(f"  {m:20s} acc={test[(m, 'accuracy')]:.3f} macroF1={test[(m, 'macro_f1')]:.3f}")

    metrics = pd.DataFrame(all_rows, columns=METRIC_COLUMNS)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    summary["notes"] = [
        "segment_future is the leakage-free headline result; segment_lifetime is optimistic because "
        "its labels are computed from transactions that include the early window.",
        "Model selection used cross-validation on the training split only; the test split was used once.",
        "Future labels use a fixed-length horizon for every customer, so acquisition date does not leak into them; "
        "customers acquired in the last window+horizon days are excluded.",
    ]
    (RESULTS / "classification_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (RESULTS / "classification_leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\nWrote results/metrics.csv, classification_summary.json, leakage audit, confusion matrices.")


if __name__ == "__main__":
    main()
