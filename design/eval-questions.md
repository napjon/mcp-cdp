# Release checklist — 100 questions

These 100 questions are the release checklist for the agreed application. Source text is preserved from the session register. **A written design alone cannot establish that a feature passes.**

## Pass / Fail / Not verified protocol

| Status | Meaning | Allowed evidence |
|---|---|---|
| **Pass** | The behavior exists **and** recorded evidence shows it meets the question | Automated test output, adversarial case, recorded observation, benchmark log, or signed usability note. Links go in **Evidence**. |
| **Fail** | The behavior was exercised **and** evidence shows it does not meet the question | Same kinds of artifacts; **Remaining work** names the fix. |
| **Not verified** | No adequate evidence yet | **Evidence** stays empty. This is the default. Design documents, plans, and unimplemented code do not move a row out of Not verified. |

Rules:

- Unverified answers remain open.
- Automated checks link to test results.
- Human usability checks link to recorded observations.
- Live integration evidence is recorded separately from test doubles (question 97).
- Self-review of `design/` does not Pass questions 5–7; those still need an independent check against implementation (and, for question 6, the original 44-item artifact).

## Evidence register

Every row starts as **Not verified**, **Evidence** empty. Keyed 1–100.

### Implementation snapshot (2026-09-13)

Automated tests: 71 passed (`pytest`). Playwright walkthrough on `http://127.0.0.1:5173` completed CSV upload → classify `survived` → train → report (macro-F1 / balanced accuracy / confusion matrix / importance) → score another CSV → download. Chat notice shown with no LLM key; no page JS errors. These observations do **not** flip rows to Pass by themselves except where a row below is updated. XGBoost did not load on this Apple Silicon machine (Intel Homebrew `libomp`); linear was selected. Live Codex/Claude Desktop MCP sessions were not exercised. The original 44-item fieldbook artifact remains missing.

### Superseded implementation snapshot (2026-09-14)

This snapshot records the state before the current fix wave. Its 111-test count and missing lifecycle items are superseded by the current snapshot below. Rows remain Not verified until evidence is written per question.

### Implementation snapshot (2026-09-14, current tree)

Measured this wave (do not treat as Pass evidence for the table below): `.venv/bin/pytest -q -p no:cacheprovider` → **157 passed**. `.venv/bin/ruff check app tests` → clean. `cd frontend && npm run typecheck` passed. `cd frontend && npm run test` → **11 passed**. After `npm ci`, frontend build and lint both pass. CONTRACT lists disclosure routes. Backup archives SQLite plus datasets/artifacts and retention runs on startup and on a maintenance loop. Still **HUMAN_ONLY**: Q3, Q6, Q31, Q81, Q100. Still **MISSING**: Q98 40-case assistant eval harness and Q99 laptop benchmark. Rows below stay Not verified until evidence is written per question.

### 1–10 Product completeness and architecture

| # | Question | Status | Evidence | Remaining work |
|---|---|---|---|---|
| 1 | Can a user complete import, configuration, training, evaluation, and prediction export through the web interface without using a terminal? | Not verified | | Implement the card path; browser walkthrough with screenshots and no terminal after start. |
| 2 | Can the entire core workflow operate when no LLM provider is configured? | Not verified | | Run import–train–predict with empty provider env; chat must explain that cards still work. |
| 3 | Can a fresh checkout be installed and started using only the documented prerequisites and commands? | Not verified | | Follow root README on a clean machine; record commands and versions. |
| 4 | Does the application run without requiring `data-platform` or any other sibling repository? | Not verified | | Install in isolation; grep imports; no sibling path in runtime. |
| 5 | Are all planned Markdown design documents present under `design/`, linked from an index, and consistent with the implementation? | Not verified | | After implementation, diff this tree against code; index is not enough. |
| 6 | Does the fieldbook document account for all 44 framework items with decisions, evidence, or explicit reasons for non-applicability? | Not verified | | Recover the original 44-item artifact or retire this gate in product. Do not invent items. |
| 7 | Do architectural decisions record alternatives, tradeoffs, owners, review dates, and conditions that would change the decision? | Not verified | | Independent review of `design/decisions.md` against this template and the running system. |
| 8 | Do REST handlers and MCP tools call shared domain services rather than implement separate business rules? | Not verified | | Code review plus tests that REST and MCP share validation and artifacts. |
| 9 | Does an automated dependency check prevent the ML engine from importing web handlers or LLM orchestration? | Not verified | | Add the `app.ml` import linter and run it in CI. |
| 10 | Does every advertised application feature perform real work, with incomplete or unavailable capabilities clearly identified? | Not verified | | Feature audit vs UI/MCP copy; no heuristic stand-ins presented as fitted models. |

### 11–20 CSV ingestion and dataset handling

| # | Question | Status | Evidence | Remaining work |
|---|---|---|---|---|
| 11 | Can users upload valid CSV files through both drag-and-drop and the file picker? | Not verified | | Frontend + API test with both input methods. |
| 12 | Are the 50 MB and 100,000-row limits enforced server-side with understandable error messages? | Not verified | | Oversize fixtures against REST and MCP; UI shows the same errors. |
| 13 | Are empty files, malformed records, and inconsistent column counts detected without creating usable partial datasets? | Not verified | | Reject fixtures; assert no `dataset_versions` row that training can use. |
| 14 | Are encoding and delimiter problems identified with a recoverable path instead of silently corrupting data? | Not verified | | Latin-1 / semicolon / mixed-newline fixtures; stored encoding/delimiter match bytes. |
| 15 | Are quoted commas, embedded newlines, escaped quotes, Unicode, and line-ending differences parsed correctly? | Not verified | | Parser tests for each case vs golden normalized table. |
| 16 | Are blank and duplicate column names resolved explicitly before users configure a model? | Not verified | | Upload headers `""` and duplicates; UI shows resolved names before Columns. |
| 17 | Are identifier values such as `00123` preserved until users explicitly choose a numeric conversion? | Not verified | | CSV with leading zeros; normalized file still has `00123`. |
| 18 | Does the preview accurately reflect the dataset that will be used, while clearly identifying any preview sampling? | Not verified | | Compare `/preview` to `normalized_path`; sampling labeled. |
| 19 | Does profiling correctly report row counts, inferred types, missing values, duplicate rows, and constant columns? | Not verified | | Profile fixture with known stats; assert `columns` rows. |
| 20 | Does every successful import create a traceable dataset version with original data, normalized data, content hash, and import metadata? | Not verified | | Inspect `dataset_versions` paths, hash, `import_meta_json`. |

### 21–30 Google Sheets connections

| # | Question | Status | Evidence | Remaining work |
|---|---|---|---|---|
| 21 | Can a sheet shared with “Anyone with the link” be imported without Google sign-in or application credentials? | Not verified | | Public sheet fixture via mocked export; no OAuth code paths. |
| 22 | Are private sheets, revoked sharing, and disabled export access reported accurately without attempting to bypass restrictions? | Not verified | | Mock 401/403 → `private`, 404 → `revoked`; no follow of off-allowlist redirects. |
| 23 | Are spreadsheet and tab identifiers parsed correctly from supported links, including query parameters and URL fragments? | Not verified | | URL parser unit tests; never fetch the user URL. |
| 24 | When a link does not identify a tab, does the interface explain how to provide the desired tab’s address? | Not verified | | Missing-gid case in UI and MCP error text. |
| 25 | Does importing a selected tab retrieve that tab’s actual values without executing spreadsheet formulas locally? | Not verified | | Export CSV mock with computed values; no formula engine in-repo. |
| 26 | Can users preview the imported sheet and select the correct header row before confirming the dataset? | Not verified | | `header_row` in request; preview before confirm. |
| 27 | Does each saved connection retain its selected tab, fetch status, last-check time, and associated dataset versions? | Not verified | | Read `sheet_connections` after import and refresh. |
| 28 | Does refreshing changed content create a new immutable version while preserving earlier models, reports, and predictions? | Not verified | | Change bytes, refresh, assert new version id; old `models.dataset_version_id` unchanged. |
| 29 | Does refreshing unchanged content avoid duplicate versions, while changed schemas require review before further use? | Not verified | | Same-hash refresh; then schema-change refresh blocks train until roles reviewed. |
| 30 | Are Google Sheets URLs, redirects, response types, timeouts, and size limits validated, with failures preserving the previous snapshot? | Not verified | | Adversarial fetch tests; previous `last_version_id` retained on failure. |

### 31–40 Problem definition and data quality

| # | Question | Status | Evidence | Remaining work |
|---|---|---|---|---|
| 31 | Does the interface distinguish predicting a category, predicting a number, and predicting future values using understandable examples? | Not verified | | Cards copy review with non-technical readers; three task examples present. |
| 32 | Must users explicitly confirm the target column before training begins? | Not verified | | Submit without target confirmation rejected on UI, REST, and MCP. |
| 33 | Can users review and override inferred numeric, categorical, text, date, identifier, and excluded-column roles? | Not verified | | PUT columns; train uses `user_role`. |
| 34 | Are identifiers and constant columns excluded by default, with the reason visible to the user? | Not verified | | Profile fixture; Columns card shows reasons. |
| 35 | Are features that may reveal the outcome or be unavailable during prediction flagged for review? | Not verified | | Leakage-flag fixtures; train blocked or warned until review. |
| 36 | Are missing targets counted and disclosed before any affected rows are excluded? | Not verified | | Report/UI count matches rows dropped; happens before fit. |
| 37 | Are invalid targets, single-class classification, and insufficient class support blocked before training? | Not verified | | Fixtures for each block; job never `running`. |
| 38 | Can users identify repeated entities so related rows are evaluated using an appropriate group-aware split? | Not verified | | `entity_column` + `split=group`; membership tests. |
| 39 | Does the training review card show the target, features, split strategy, budget, and relevant data-quality warnings? | Not verified | | Screenshot of Confirm & train vs frozen `config_json`. |
| 40 | Does each submitted job preserve an immutable experiment revision, even if users later change their configuration? | Not verified | | Submit, edit config, assert old `experiment_revisions` row unchanged. |

### 41–50 Classification, regression, and model selection

| # | Question | Status | Evidence | Remaining work |
|---|---|---|---|---|
| 41 | Does preprocessing handle missing numeric values and apply scaling only where the selected estimator needs it? | Not verified | | Inspect fitted pipeline; linear scaled, trees not. |
| 42 | Does categorical preprocessing handle missing values and unseen categories while bounding encoded feature growth? | Not verified | | Unseen-level predict test; encoded width cap test. |
| 43 | Does text preprocessing use bounded TF-IDF features fitted exclusively on the relevant training partition? | Not verified | | Vocabulary/idf identity vs train-only refit. |
| 44 | Does candidate comparison include a dummy baseline, a regularized linear model, and XGBoost? | Not verified | | Metrics payload lists all three (or documented inapplicable). |
| 45 | Are preprocessing, feature selection, and hyperparameter search isolated from validation folds and the final test set? | Not verified | | Leakage tests: preprocessor fit on train only. |
| 46 | Is an untouched 20% test partition reserved where the configured split is valid, with no silent fallback to an invalid split? | Not verified | | Invalid split rejected; test ids persisted. |
| 47 | Are candidate models selected using validation performance rather than their final test scores? | Not verified | | Winner has better validation, not necessarily better test. |
| 48 | Are random seeds, split membership, parameters, dependency versions, and candidate failures recorded sufficiently to reproduce a run? | Not verified | | Replay job from stored metadata; compare metrics. |
| 49 | Are Quick and Thorough budgets enforced, including final fitting and evaluation, without publishing incomplete models? | Not verified | | Abort mid-search leaves `failed` and no ready model. |
| 50 | Do classification and regression metrics match an independent recomputation from stored actuals and predictions? | Not verified | | `tests/test_ml*` recomputes macro-F1 / MAE / RMSE / R². |

### 51–60 Grouped time-series forecasting

| # | Question | Status | Evidence | Remaining work |
|---|---|---|---|---|
| 51 | Can users configure a date column, numeric target, and optional grouping columns such as SKU and store? | Not verified | | UI + REST + MCP config round-trip. |
| 52 | Are daily, weekly, and monthly frequencies validated, with ambiguous date parsing requiring user correction? | Not verified | | Ambiguous date fixture blocked; D/W/M accepted. |
| 53 | Are the limits of 100 series and a 1–90-period horizon enforced consistently in the UI, REST API, and MCP tools? | Not verified | | 101 series and horizon 0/91 rejected on all three surfaces. |
| 54 | Are duplicate timestamps detected within each group and resolved through an explicit user decision? | Not verified | | `reject` vs `aggregate_mean`; no silent keep. |
| 55 | Are missing periods reported without silently treating missing demand as zero? | Not verified | | Gap fixture; zeros not imputed by default. |
| 56 | Are insufficient-history groups identified individually and excluded only after the user reviews the affected subset? | Not verified | | Per-group list on the review card; exclusion after confirm. |
| 57 | Does forecasting compare last-value and seasonal-naive baselines with the pooled XGBoost candidate wherever those candidates are applicable? | Not verified | | Report lists applicable candidates and selection source. |
| 58 | Are lag and rolling features computed within each series without crossing group boundaries or using future values? | Not verified | | Group-isolation and no-future-leak tests. |
| 59 | Do rolling-origin backtests and the final untouched horizon reproduce actual forecasting conditions, including recursive prediction? | Not verified | | Recursive multi-step test vs stored horizon. |
| 60 | Are unseen groups and unsupported forecasting configurations rejected clearly, with no uncalibrated uncertainty bands presented as reliable? | Not verified | | Unseen group predict error; no fake intervals in UI. |

### 61–70 Reports, explanations, and prediction reuse

| # | Question | Status | Evidence | Remaining work |
|---|---|---|---|---|
| 61 | Does every metric identify its dataset version, evaluation split, sample count, and units? | Not verified | | Inspect `report_json` / `metrics_json` keys. |
| 62 | Do classification reports show macro-F1, balanced accuracy, per-class precision/recall, and a correctly labeled confusion matrix? | Not verified | | Plot labels match class order; numbers match Q50 recompute. |
| 63 | Do regression reports show MAE, RMSE, R², and residual plots with explicit handling of undefined metrics? | Not verified | | Undefined R² case explained, not numeric fiction. |
| 64 | Do forecasting reports show aggregate and per-series errors, baseline comparisons, and aligned actual-versus-predicted dates? | Not verified | | Per-series table + date-aligned plot. |
| 65 | Are validation and test results visibly separated, with accurate comparisons against the baseline even when performance is worse? | Not verified | | Winner-worse-than-dummy fixture still shows dummy. |
| 66 | Are feature-importance calculations tied to the recorded model and evaluation sample, and are probabilities and explanations presented without causal or certainty claims? | Not verified | | Copy review; importance recomputed from stored model. |
| 67 | Do saved models produce consistent predictions after application restart using the original fitted preprocessing pipeline? | Not verified | | Train, restart process, predict; byte-stable scores. |
| 68 | Are prediction columns matched by name, with missing required fields and conversion failures reported while extra columns are handled consistently? | Not verified | | Missing/extra/bad-type predict fixtures. |
| 69 | Are prediction rows kept in input order, with identifiers preserved and spreadsheet formulas escaped safely in downloaded text cells? | Not verified | | Order + `00123` + leading `=` escaped in download. |
| 70 | Can users optionally supply actual outcomes in new data and receive a separate evaluation without refitting the saved model? | Not verified | | Predict-with-actuals job; `models.artifact_path` mtime unchanged. |

### 71–80 Built-in chat and provider integration

| # | Question | Status | Evidence | Remaining work |
|---|---|---|---|---|
| 71 | Can both OpenAI and Anthropic integrations authenticate, stream responses, report failures, and record the selected provider and model? | Not verified | | Recorded live runs per provider, stored separately from doubles. |
| 72 | Are API keys kept server-side and excluded from browser storage, SQLite records, logs, and error responses? | Not verified | | Grep artifacts; forced provider error body; `/api/status` has no secrets. |
| 73 | Are user messages persisted before generation, with partial, completed, interrupted, and failed assistant messages represented accurately? | Not verified | | Kill stream mid-flight; inspect `messages.status`. |
| 74 | Does reconnecting to chat recover persisted state without duplicating messages or silently restarting completed actions? | Not verified | | `client_id` replay + refresh during `complete`. |
| 75 | Is the default provider context limited to approved schema, aggregates, experiment settings, and computed results? | Not verified | | Capture outbound payload with disclosure off. |
| 76 | Does sharing raw samples require an explicit per-dataset choice and preview, with revocation affecting subsequent requests? | Not verified | | Enable, preview, send; revoke; assert later payload has no rows. |
| 77 | Are tool arguments schema-validated, limited to the active project, and constrained to six tool calls per turn? | Not verified | | Cross-project id rejected; seventh tool call refused. |
| 78 | Can the assistant propose configuration changes without starting training until the user acts on the concrete review card? | Not verified | | Chat-only config leaves jobs empty until Confirm & train. |
| 79 | Are metric claims grounded in persisted results, with unavailable evidence acknowledged rather than replaced by invented values? | Not verified | | Ask for metrics with no report; assistant must not invent. |
| 80 | Are context/output limits enforced and provider failures handled while recording actual usage and honestly labeled cost estimates? | Not verified | | Overlong prompt; failed provider; `usage_json` vs labeled estimate. |

### 81–90 MCP compatibility and access boundaries

| # | Question | Status | Evidence | Remaining work |
|---|---|---|---|---|
| 81 | Do documented connection procedures work with Codex, Claude Code, and Claude Desktop? | Not verified | | Follow `design/chat-and-mcp.md` on each client; record tool list. |
| 82 | Does the server support Streamable HTTP and a stdio bridge without each connection starting a separate worker or database? | Not verified | | Two clients attached; one `jobs` runner; one SQLite file. |
| 83 | Are tool names, descriptions, input schemas, output schemas, and read/write annotations accurate and discoverable? | Not verified | | `tools/list` vs contract table. |
| 84 | Can an MCP client list projects, inspect datasets, read profiles, and retrieve model reports? | Not verified | | Read-tool integration test. |
| 85 | Can an MCP client connect and refresh a public Google Sheet using the same validation and versioning rules as the web interface? | Not verified | | Same mocks as Q21–30 through MCP. |
| 86 | Can MCP clients configure experiments, submit training, inspect progress, cancel jobs, and run batch predictions? | Not verified | | Write tools with `confirm: true`; cancel during run. |
| 87 | Do equivalent REST and MCP operations produce equivalent configurations, validation outcomes, and artifact contents? | Not verified | | Pairwise hash of artifacts and error codes. |
| 88 | Do long-running MCP operations return job identifiers promptly, with bounded summaries and usable artifact references? | Not verified | | Submit returns `{job_id}` before fit finishes. |
| 89 | Does the stdio bridge restrict file imports to configured directories, including protection against traversal and symlink escapes? | Not verified | | `import_local_file` adversarial paths. |
| 90 | Are MCP requests authenticated and project-scoped, with the same sample-disclosure policy and no tools for arbitrary shell, SQL, code execution, or unrestricted URL fetching? | Not verified | | Missing token 401; tool inventory audit; disclosure flag honored. |

### 91–100 Reliability, security, and release evidence

| # | Question | Status | Evidence | Remaining work |
|---|---|---|---|---|
| 91 | Are SQLite foreign keys, migrations, WAL configuration, and transactional job claims verified without holding transactions during training? | Not verified | | PRAGMA checks; long-train with concurrent reads; claim tests. |
| 92 | Are one active ML subprocess and ten queued jobs enforced, with visible queue saturation and responsive control requests? | Not verified | | Eleventh submit errors; cancel during the running job. |
| 93 | Do leases, heartbeats, and attempt identifiers prevent duplicate execution or stale workers from publishing results? | Not verified | | Kill worker mid-job; stale lease must not overwrite succeeded artifacts. |
| 94 | Do cancellation, process termination, application restart, navigation, and event reconnection preserve consistent jobs and artifacts? | Not verified | | Crash/restart matrix; SSE reconnect. |
| 95 | Do backup, restore, project deletion, and thirty-day trace retention behave correctly without exposing secrets or deleting unrelated files? | Not verified | | Copy `data/`; restore; deletion cascade; retention job; no keys in backup of SQLite. |
| 96 | Are loopback binding, session authentication, CSRF protection, host/origin checks, and private-network fetch restrictions verified by failing adversarial cases? | Not verified | | Bind test; CSRF without header; Sheets allowlist bypass attempts. |
| 97 | Do unit, integration, browser, and adversarial suites exercise both ingestion sources and all three prediction tasks, with live integration evidence recorded separately from test doubles? | Not verified | | Suite map + a live-evidence folder distinct from mocks. |
| 98 | Does the forty-case assistant evaluation run three times per release model, meet 95% correct workflow/tool decisions, and observe zero fabricated metrics or prohibited disclosures? | Not verified | | Run the 40-case harness three times; store transcripts and scores. |
| 99 | On the recorded reference laptop, does the 10,000-row Quick benchmark finish within five minutes while control API p95 remains below 500 ms during training? | Not verified | | Benchmark log: hardware, wall time, API latency histogram. |
| 100 | Do four of five non-technical users finish import-to-prediction unaided within ten minutes of active interaction, identify the model’s main limitation, and have their findings reflected in release evidence? | Not verified | | Five-person session notes; 4/5 criterion; limitations filed back into reports/copy. |
