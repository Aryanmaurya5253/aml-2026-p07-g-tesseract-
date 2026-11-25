# P07 — Customer Segmentation for an Online Retailer

**Course:** Applied Machine Learning (CSAI2017P), Autumn 2026  
**Theme:** P07 — Retail Segmentation  
**Group:** Team Tesseract

## Project question
Which genuinely distinct customer groups exist in online retail transactions, and can a model assign a new customer to a segment using only their early purchase behaviour?

## Dataset
Use [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii). Download the dataset ZIP in a browser, extract the workbook, and place it at `data/raw/online_retail_II.xlsx`. See [dataset/README.md](dataset/README.md) for the download and placement instructions.

**Do not commit the raw ZIP or workbook.** The workbook is approximately 43.5 MB and remains local; the repository tracks the source, instructions, scripts and analysis outputs.

## Setup
Recommended Python 3.11 or 3.12. Use a consistent Python version across the team.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Phase 1 — Dataset loading and cleaning

The workbook was inspected and both sheets were combined.

| Source sheet | Rows | Columns |
|---|---:|---:|
| Year 2009–2010 | 525,461 | 8 |
| Year 2010–2011 | 541,910 | 8 |
| **Combined** | **1,067,371** | **8** |

The current baseline retains **776,672 transactions (72.765%)** after sequential cleaning. See [cleaning funnel](results/cleaning_funnel.csv) and [cleaning summary](results/cleaning_summary.json).

Cleaning decisions:
- Remove exact duplicate rows.
- Remove transactions without Customer ID because customer-level RFM requires customer identity.
- Exclude cancellation invoices whose Invoice value begins with `C`; these are not netted against original purchases.
- Exclude the explicit non-product/service StockCode list in the cleaning script.
- Remove rows with invalid core fields and non-positive quantities or prices.
- Do not automatically cap or remove extreme positive quantity/price values; inspect them during EDA.

Run from the repository root:

```bash
python src/clean_retail_data.py
```

This creates the local file `data/processed/transactions_clean.csv` and the Phase 1 diagnostics. The raw and processed transaction data are excluded from Git.

## Phase 2 — Exploratory Data Analysis

The EDA script calculates monthly revenue, product performance, country-level activity, missing-value checks and IQR-based outlier diagnostics.

```bash
python src/run_eda.py
```

The script reads `data/processed/transactions_clean.csv` and writes:
- [Monthly revenue](results/monthly_revenue.csv)
- [Top 10 products by revenue](results/top_10_products.csv)
- [Country analysis](results/country_analysis.csv)
- [Missing values after cleaning](results/missing_values_cleaned.csv)
- [Outlier summary](results/outlier_summary.csv)
- [EDA summary](results/eda_summary.json)
- Customer purchasing patterns: `customer_purchasing_patterns.csv` (one row per customer, generated locally), [purchase frequency buckets](results/purchase_frequency_buckets.csv), weekday and hourly activity, and monthly new-vs-returning customers.
- Six charts under `results/figures/` (monthly revenue with partial months marked, top products, top countries, invoices per customer, weekday revenue, new vs returning customers).

Changes after review: products are grouped by `StockCode` (the most frequent description is shown) so a product with several description spellings is not split; the first/last months are flagged `is_partial_month` and excluded from the peak/low month statistics; the EDA summary now includes a `purchasing_patterns` block. Re-run `python src/run_eda.py` and re-check the figures quoted below.

### Initial EDA findings

- Cleaned transactions: **776,672**
- Unique customers: **5,861**
- Unique invoices: **36,669**
- Distinct stock codes: **4,626**
- Countries: **41**
- Period covered: **1 December 2009 – 9 December 2011**
- Gross transaction revenue: **£17,073,528.31**
- Highest-revenue month: **November 2010 (£1,156,008.39)**
- Highest-revenue country: **United Kingdom (£14,291,644.57)**

The highest-revenue product is **REGENCY CAKESTAND 3 TIER (£277,656.25)**. A separate product, **PAPER CRAFT, LITTLE BIRDIE**, has 80,995 units recorded in one invoice and revenue of £168,469.60. This unusual observation is flagged for investigation, not automatically removed.

Outliers are flagged using the Tukey upper fence (Q3 + 1.5 × IQR). These flags are diagnostic only; no upper-tail observations were removed in Phase 2.


## Phase 3 — Customer-level RFM analysis

The RFM pipeline builds one row per customer from the cleaned transactions.

Run:

```bash
python src/build_rfm.py
```

Definitions:
- **Recency:** days from the fixed reference date to the customer's most recent purchase date; lower is more recent.
- **Frequency:** number of distinct invoice IDs per customer (not transaction-line count).
- **Monetary:** sum of Quantity × Price across retained transactions, in GBP.

The reference date is **10 December 2011**, one calendar day after the latest retained transaction date (9 December 2011). This is fixed from the dataset rather than the machine's current date.

### RFM results

- Customers represented: **5,861**
- Average Recency: **201.35 days**; median: **96 days**
- Average Frequency: **6.26 distinct invoices**; median: **3**
- Average Monetary: **£2,913.07**; median: **£854.99**

Raw feature skewness is substantial for Frequency (**12.12**) and Monetary (**25.34**). The pipeline applies `log1p` to each RFM feature and then `StandardScaler` for a candidate clustering matrix. After log transformation, skewness is **-0.45** for Recency, **1.00** for Frequency and **0.27** for Monetary. This reduces skew, especially for Monetary; Frequency remains right-skewed.

Outputs generated locally:
- `results/customer_rfm.csv` — customer-level RFM table
- `results/rfm_scaled.csv` — standardized log-RFM matrix for clustering
- `results/rfm_feature_statistics.csv` — raw feature statistics
- `results/rfm_log_feature_statistics.csv` — log-transformed feature statistics
- `results/rfm_summary.json` — definitions, reference date and summary
- `results/figures/rfm_*.png` — raw and transformed distributions

The two customer-level CSVs are generated locally and are not committed; run the script to recreate them. Summary statistics are tracked in Git.

**Interpretation caveat:** Monetary represents gross retained sales because cancellation invoices were excluded in Phase 1, not netted against original purchases. Customer ID is an identifier and is not included as a clustering feature.


## Phase 4A — K-Means baseline

The K-Means baseline evaluates cluster counts from **k=2 through k=10** using the three standardized log-RFM features: Recency, Frequency and Monetary. The run uses `random_state=42`, `n_init=20` and the Lloyd algorithm.

Run from the repository root after Phase 3:

```bash
python src/clustering.py
```

### Evaluation design

- **Inertia:** calculated using all 5,861 customers; lower values are expected as k increases.
- **Silhouette:** `silhouette_sampled` uses a deterministic sample of 2,000 customers (kept so the originally recorded numbers stay comparable); `silhouette_full` uses all customers and is the value used for later decisions. `min_cluster_share` records the smallest cluster's share.
- **Davies-Bouldin and Calinski-Harabasz:** calculated using all customers.
- Customer IDs are retained only to map cluster assignments back to customers; they are not model features.

### Baseline result

The highest sampled silhouette among tested values was at **k=2 (0.4444)**. For k=2, the Davies-Bouldin index was **0.8730** and the Calinski-Harabasz score was **6177.7656**. This makes k=2 a provisional candidate, not a final decision. We must also inspect cluster sizes, customer RFM profiles, the elbow curve and business usefulness before selecting a final segmentation.

See [K-Means metrics](results/kmeans_metrics.csv) and [K-Means summary](results/kmeans_summary.json). The script also generates customer assignments, cluster profiles and elbow/silhouette/PCA plots locally under `results/` and `results/figures/`.

## Phase 4B — Hierarchical clustering, DBSCAN and stability

```bash
python src/clustering_advanced.py
```

- **Hierarchical:** Ward, complete and average linkage cut at k=2–10 (silhouette on all customers, Davies-Bouldin, Calinski-Harabasz, cophenetic correlation), plus truncated dendrograms for Ward and average linkage, and the Adjusted Rand Index against K-Means (`results/hierarchical_metrics.csv`, `hierarchical_vs_kmeans_ari.csv`).
- **DBSCAN:** k-distance plot, an (eps, min_samples) grid, and noise analysis. A configuration is accepted only with at least two clusters, each holding at least 1% of customers, and at most 15% noise; if none qualifies that is reported as a finding (`results/dbscan_grid.csv`, `dbscan_noise_profile.csv`).
- **Stability:** K-Means ARI across 20 random 80% subsamples for every k (`results/kmeans_stability.csv`).
- `results/clustering_comparison.json` and `algorithm_comparison.csv` summarise all three algorithms for k = 3–6.

## Phase 5 — Segments and recommendations

```bash
python src/segments.py          # k chosen by the documented rule
python src/segments.py --k 4    # or override it
```

The final k is chosen by an explicit rule (`results/k_selection.csv`): among k = 3–6, the highest full-data silhouette whose smallest segment holds at least 5% of customers and whose stability ARI is at least 0.75. k = 2 is reported but not chosen automatically because it can only separate active from lapsed customers. Segment names come from heuristic thresholds on the standardised centroids and must be reviewed by the team; override with `--k` if the profiles say otherwise.

Outputs: `segment_profiles.csv` (sizes, revenue share, RFM medians), `segment_recommendations.csv`, `segment_summary.json`, a local `customer_segments.csv`, and figures (PCA projection, RFM profile, customer share vs revenue share).

## Phase 6 — Classification from early behaviour (leakage-safe)

```bash
python src/classification.py                       # both label sources
python src/classification.py --window-days 60 --label-source future
```

- **Features:** only each customer's first `--window-days` (default 30) after their first purchase: number of invoices, lines, units, spend, average and maximum invoice value, first-invoice value, distinct products, active days, day span, average unit price, UK flag, weekday and hour of the first purchase. Lifetime Recency/Frequency/Monetary and any acquisition date/month are excluded.
- **Labels:** `future` (headline) = K-Means segments built only from the following `--horizon-days` (default 90), so labels and features share no transactions and acquisition date cannot leak into the labels; `lifetime` = the Phase 5 segments, which are computed from all transactions including the early window, so that result is optimistic and reported for comparison only.
- **Hygiene:** customers need a complete early window and horizon; scaling/log transforms are inside Pipelines fitted on the training split only; hyper-parameters are tuned by cross-validation on the training split; the test split is used once; classes with fewer than 20 customers are dropped and reported. A leakage audit is written to `results/classification_leakage_audit.json`.
- **Models:** majority baseline, Logistic Regression, Decision Tree, Random Forest.

## Phase 7 — Evaluation, report and submission

`results/metrics.csv` (columns `model, task, split, metric, value, notes`) holds accuracy, macro-F1, weighted-F1, balanced accuracy and per-class recall/precision for every model and task, plus cross-validated macro-F1 on the training split. Confusion matrices are written to `results/confusion_matrix_*.csv` and `results/figures/confusion_matrix_*.png`.

```bash
python src/make_report.py   # drafts report/report.md from the files in results/
```

The report generator only fills in numbers and data-driven sentences; the team must read, edit and shorten the draft to 4–6 pages and export it to PDF.

## Run everything

```bash
python src/run_all.py                # all steps (needs data/raw/online_retail_II.xlsx)
python src/run_all.py --from-step 3  # resume from RFM
python tests/smoke_test.py           # end-to-end check on SYNTHETIC data (never touches results/)
```

Optional sensitivity check for the cancellation decision: `python src/clean_retail_data.py --drop-reversed-orders` also drops purchase lines that a later cancellation line reverses (same customer, stock code and quantity). It is off by default so the documented baseline is unchanged; if you adopt it, regenerate and re-record every downstream result.

## Pipeline overview
| Phase | Script | Main outputs |
|---|---|---|
| 1 Cleaning | `src/clean_retail_data.py` | `results/cleaning_funnel.csv`, `cleaning_summary.json` |
| 2 EDA | `src/run_eda.py` | monthly/product/country tables, purchasing patterns, figures |
| 3 RFM | `src/build_rfm.py` | `rfm_summary.json`, feature statistics |
| 4 Clustering | `src/clustering.py`, `src/clustering_advanced.py` | K-Means, hierarchical, DBSCAN, stability metrics |
| 5 Segments | `src/segments.py` | `segment_profiles.csv`, recommendations, PCA plot |
| 6-7 Classification and evaluation | `src/classification.py` | `metrics.csv`, confusion matrices, leakage audit |
| 7 Report | `src/make_report.py` | `report/report.md` (draft) |

## Team
| Member | Enrolment number | GitHub username |
|---|---:|---|
| Aryan Kumar Maurya | 590022160 | TODO |
| Anirudh Sarkar | 590025468 | TODO |
| Devam Yadav | 590022492 | TODO |
| Aniket Malik | 590024567 |MALIK87ANIKET|
| Mansha | 590021940 | TODO |
| Shatakshi | 590021766 | TODO |
| Saisradha Das | 590029397 | TODO |
| Pragati | 590028725 | TODO |

## Contributions
Agree and record actual responsibilities here. Every member should understand the code and decisions they contribute.

## AI and external resources
Claude (Anthropic) was used to help write and test the scripts for Phase 2 additions, Phases 4B–7, the notebooks and the smoke test, and to review the earlier phases. Each member must read and understand the code they submit (the individual assessment tests understanding). Data: UCI Online Retail II. Libraries: pandas, NumPy, scikit-learn, SciPy, Matplotlib, openpyxl.

## Course milestones
- `checkpoint-1`: actual data loaded, shape/head/chart, RFM table, and one baseline K-Means run. This is complete; create the annotated tag (`git tag -a checkpoint-1 -m "..."`) when the team agrees.
- `checkpoint-2`: full clustering pipeline and initial results. Run `python src/run_all.py` on the real data, review the outputs, then tag.
- Final: analysis, evaluation, report, README and `REPRODUCIBILITY.md`.

## Headline results
Not yet recorded: the Phase 4B–6 scripts have been tested end-to-end on synthetic data only. Run `python src/run_all.py` on the real workbook, review `results/segment_profiles.csv`, `results/k_selection.csv` and `results/metrics.csv`, then copy the real numbers here. Cleaning retention and EDA statistics are data-preparation results, not model-performance metrics.

## Current status
Phases 1–3 and the K-Means baseline (4A) were run on the real data. Phases 2 (purchasing patterns), 4B, 5, 6 and 7 are implemented and pass an end-to-end smoke test on synthetic data and under the pinned `requirements.txt`; they still need to be run on the real workbook, their outputs reviewed (especially segment names and the chosen k), and the report edited.
