# Data and ML

How mcp-cdp ingests tables, assigns roles, trains, scores, and stores artifacts. REST, MCP, and the worker share these rules. `app.ml` implements `run_train` / `run_predict` only; it does not import web handlers or LLM clients.

Sibling product contrast: `data-platform` prediction paths score with LLM structured outputs and stdlib heuristics (token overlap, lexicons, simple numeric fallbacks). This product **fits** dummy, regularized linear, and XGBoost pipelines on the operator's table, persists the fitted objects, and reuses them for prediction after restart.

## Ingest

### CSV

- Entry: `POST /api/projects/{id}/datasets/upload` (multipart `file`) or MCP `import_local_file` (stdio only, configured directories, reject traversal and symlink escapes).
- Server enforces **50 MB** and **100,000 rows**. Oversize is an error, not a silent sample.
- Empty files, malformed records, and inconsistent column counts do not become usable datasets.
- Encoding and delimiter are detected and stored (`dataset_versions.encoding`, `delimiter`). Recoverable failure beats silent corruption.
- Quoted commas, embedded newlines, escaped quotes, Unicode, and mixed line endings must parse.
- Blank and duplicate headers are resolved **before** the operator configures a model, with the resolution recorded in `import_meta_json`.
- Values such as `00123` stay text until the operator explicitly marks the column numeric.
- Preview (`n=20` by default) is labeled as a preview. It must match the normalized table that training will read.

### Google Sheets

- Entry: `POST /api/projects/{id}/datasets/sheets` `{url, gid?, header_row?, name?}`.
- Sharing required: **Anyone with the link**. No Google user OAuth.
- **Never fetch the user URL.** Parse spreadsheet id and gid, then GET  
  `https://docs.google.com/spreadsheets/d/{id}/export?format=csv&gid={gid}`.
- Allowlist hostnames: `docs.google.com`, `spreadsheets.google.com`, `*.googleusercontent.com`.
- Login wall / 401 / 403 → connection status `private`. 404 → `revoked`.
- If the link does not identify a tab, the UI/MCP error explains how to copy the tab's address (gid).
- Export returns computed values; this app does not evaluate spreadsheet formulas.
- Header row is selectable before confirm.
- Hash canonical CSV bytes. Same hash → refresh does not insert a version. Changed bytes → new immutable `dataset_versions` row; old models/reports/predictions keep their version ids.
- Schema change after refresh requires the operator to review roles before the next train.
- Timeouts, redirects, non-CSV content types, and size limits fail without deleting the previous snapshot.
- Refresh is **manual**: `POST .../sheet-connections/{id}/refresh` or MCP `refresh_google_sheet`.

### Versions

Every successful import inserts `dataset_versions` with `original_path`, `normalized_path`, `content_hash`, `n_rows`, `n_cols`, `header_row`, `encoding`, `delimiter`, `import_meta_json`. Originals are kept so a later parse bug can be diagnosed against the bytes the operator actually sent.

## Profiling and roles

Profiler writes `columns`:

| Field | Meaning |
|---|---|
| `inferred_role` | `numeric` / `categorical` / `text` / `date` / `identifier` / `excluded` |
| `user_role` | Operator override; wins at train time |
| `n_missing`, `n_unique`, `is_constant` | Quality stats |
| `sample_preview` | For the UI; LLM sees this only with disclosure |

Rules:

- Identifiers and constant columns are **excluded by default**, with the reason shown on the Columns card.
- Operator can override inferred roles via `PUT .../datasets/{id}/columns`.
- Features that may leak the target or that would be unavailable at prediction time are flagged for review (not auto-dropped unless constant/identifier default applies).
- Missing targets are counted and disclosed **before** those rows are dropped.
- Invalid targets, single-class classification, and classes below a documented support minimum block training.
- Repeated entities can be named (`entity_column`) so split=`group` keeps related rows together.

## Experiment config

Frozen on submit as `experiment_revisions.config_json`. Shape is the contract JSON:

- `task`: `classification` | `regression` | `forecast`
- `target`, `features`, `excluded`, `roles`
- `group_columns` (forecast: product + location, e.g. sku + store)
- `date_column`, `frequency` `D|W|M`, `horizon` 1–90
- `duplicate_timestamp`: `reject` | `aggregate_mean`
- `entity_column`, `split` `random` | `group` | `time`, `test_size` 0.2
- `budget` `quick` | `thorough`, `seed` 42

The Confirm & train card must show target, features, split, budget, and outstanding quality warnings. Submit always creates a new revision; editing the card later does not rewrite a submitted one.

## Holdout by task (do not mix these)

The experiment JSON contains both `test_size` and `horizon`. They are not interchangeable.

| Task | Holdout | Invalid and rejected |
|---|---|---|
| Classification, regression | Untouched **20%** (`test_size: 0.2`) using `split` `random` \| `group` \| `time` | Silent fallback to another split; shrinking test to “whatever is left” |
| Forecast | Final untouched **horizon** periods at `frequency` `D\|W\|M`, per series after duplicate/missing handling | Using `test_size` 0.2 as the forecast holdout; `split=random` or treating SKU+store groups as i.i.d. rows |

Forecast still uses rolling-origin **validation** on earlier history, then reports the horizon holdout once for the selected candidate. `test_size` in a forecast config is ignored for partitioning; ML must not silently apply a 20% row split to a series table.

## Shared leakage rules

These apply to all three tasks:

1. **Hold out first**, using the table above. There is no silent fallback to a different split or a different holdout definition.
2. **Fit on train only.** Imputation, scaling, category encoding, TF-IDF, lag/rolling stats, and any feature selection are fit on the training partition (and, for HPO, on inner folds / rolling-origin windows), never on the final holdout.
3. **Select on validation.** Dummy vs linear vs XGBoost (or forecast last-value / seasonal-naive vs pooled XGBoost) is chosen by validation metric. Holdout metrics are computed once for the winner and reported as test (classification/regression) or horizon (forecast).
4. **No target in features.** Target name cannot appear in `features`.
5. **Record enough to rerun:** seed, split membership (or series cut dates), parameters, dependency versions, and candidate failures.
6. **Do not publish incomplete models.** If Quick or Thorough aborts, status is `failed` and `models` is not presented as ready.

## Classification and regression

### Preprocessing

- Numeric: missing-value imputation. Scaling only when the estimator needs it (regularized linear: yes; trees: no).
- Categorical: missing as its own level; unseen categories at predict time handled without crash; encoded width bounded (frequency cutoff or hashing — pick one and persist it in the pipeline).
- Text: bounded TF-IDF, fitted only on the training partition of that candidate.

### Candidates (Quick and Thorough)

| Candidate | Role |
|---|---|
| Dummy | Baseline (most frequent / mean-median as appropriate) |
| Regularized linear | Logistic or ridge/elastic-net family |
| XGBoost | Tree ensemble |

Thorough uses the same three families with a larger hyperparameter search. It is not a different product and does not add extra estimator types.

### Metrics

Every metric names **dataset version**, **split**, **sample count**, and **units**. Validation and test are labeled separately. Comparisons to dummy remain visible when the winner is worse than dummy.

**Classification:** macro-F1, balanced accuracy, per-class precision/recall, confusion matrix with class labels in the right order.

**Regression:** MAE, RMSE, R². Residual plots. If a metric is undefined, the report says so rather than printing a fabricated number.

An independent recomputation from stored actuals and predictions must match the published numbers (eval question 50).

## Grouped forecasting

Intended shape: one numeric target, one date column, optional group columns for **product and location**.

- Frequencies: day / week / month, validated. Ambiguous dates require operator correction; do not guess.
- UI, REST, and MCP all enforce **100 series** and **horizon 1–90**.
- Duplicate timestamps inside a group: operator chooses `reject` or `aggregate_mean`. No silent keep-last.
- Missing periods are reported. Missing demand is **not** filled with zero unless the operator explicitly chooses that (default: do not).
- Groups with too little history are listed individually and excluded only after review.
- Unseen groups at predict time are rejected clearly.
- Do not draw uncalibrated uncertainty bands as if they were reliable intervals.
- Holdout is the configured **horizon**, not 20% of rows (see Holdout by task).

### Candidates

When applicable: **last-value**, **seasonal-naive**, **pooled XGBoost**. Selection still uses validation / rolling-origin, not the final holdout.

### Leakage unique to series

- Lags and rolling statistics are computed **inside each group**.
- Shifts use past values only (no current-horizon target).
- No feature may cross group boundaries.
- Rolling-origin backtest and the final untouched horizon must match production conditions, including **recursive** multi-step prediction.

### Forecast metrics

Aggregate error plus **per-series** error. Baseline comparison required. Plots align actual vs predicted on dates. Same dataset-version / split / n / units rule as the other tasks.

## Predict path

`run_predict` loads the **original fitted pipeline** from `models.artifact_path`. Restart of the app must not change scores for the same input bytes.

- Columns matched **by name**.
- Missing required fields and conversion failures are errors.
- Extra columns: ignored consistently (documented in the report).
- Row order preserved.
- Identifier strings preserved.
- Downloaded text cells formula-escaped so spreadsheet software does not execute them.
- Optional actuals in new data produce a **separate** evaluation job/metrics without refitting.

## Artifacts

| Location | Contents |
|---|---|
| `data/artifacts/<job_id>/` | Fitted pipeline (`model_path`), metrics payload, prediction file when type=`predict` |
| `data/artifacts/<job_id>/plots/` | Confusion matrix, residuals, forecast actual-vs-pred, and other figures referenced by the report |
| `models.metrics_json` | Selected candidate, validation and test metrics, candidate failures |
| `reports.report_json` | Operator-facing report body |
| `predictions.output_path` | Scored file |

`run_train` / `run_predict` return `metrics`, `model_path`, `plot_files`, `warnings`, `selected_candidate`. The worker copies those into SQLite and disk. Feature importance, when shown, is tied to the recorded model and evaluation sample. Probabilities and explanations are not causal claims and not certainty claims.

## What the worker passes to ML

`job` includes `id`, `type`, `config` (the frozen experiment config), `dataset_normalized_path`, and optional `predict_path`. `paths` includes `artifact_dir`. ML reads only those files plus its own writes under `artifact_dir`.
