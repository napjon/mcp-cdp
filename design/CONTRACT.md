# Implementation contract

Frozen split for parallel work. Do not invent a second schema or a second API.

## Product lock

- Local single-user app. Bind `127.0.0.1:8765` only. Never ports 3000 or 8000.
- FastAPI + SQLite. Self-contained. No `data-platform` imports.
- Chat + guided cards. Core workflow works with no LLM key.
- MCP clients control this app (Codex / Claude Code / Claude Desktop).
- CSV upload and public Google Sheets (`Anyone with the link`). Manual refresh.
- Tasks: classification, regression, grouped forecasting (SKU + store).
- AutoML Quick budget: dummy baseline + regularized linear + XGBoost.
- LLM sees summaries only unless the user opts into sample sharing.
- Never use the word `DEMO` in UI, docs, or code comments.
- Do not git add, commit, or open PRs.

## Directory ownership

| Owner | Paths |
|---|---|
| Design | `design/**` except this file |
| Platform | `app/settings.py`, `app/db.py`, `app/models.py`, `app/worker.py`, `app/main.py`, `app/services/{datasets,sheets,experiments,jobs,predict,security}.py`, `app/api/{projects,datasets,experiments,jobs,predict,health}.py`, `README.md`, `.env.example`, `tests/test_ingest*.py`, `tests/test_sheets*.py`, `tests/test_jobs*.py`, `tests/test_security*.py` |
| ML | `app/ml/**`, `tests/test_ml*.py` |
| Frontend | `frontend/**` |
| Chat+MCP | `app/mcp/**`, `app/services/chat.py`, `app/api/chat.py`, `tests/test_chat*.py`, `tests/test_mcp*.py` |

Do not edit files you do not own. If you need a hook, use the interfaces below.

## Runtime layout

- SQLite file: `data/mcp_cdp.sqlite` (WAL, foreign keys on).
- Uploads / originals: `data/datasets/<dataset_id>/`.
- Fitted artifacts: `data/artifacts/<job_id>/`.
- Plots: `data/artifacts/<job_id>/plots/`.

## Schema (SQLite)

Use these tables and column names. JSON is stored as TEXT.

```
projects(id TEXT PK, name TEXT, created_at TEXT)
datasets(id TEXT PK, project_id TEXT, name TEXT, source_type TEXT, /* csv|sheets */
         connection_id TEXT NULL, created_at TEXT)
dataset_versions(id TEXT PK, dataset_id TEXT, version INTEGER, content_hash TEXT,
         original_path TEXT, normalized_path TEXT, n_rows INTEGER, n_cols INTEGER,
         header_row INTEGER, encoding TEXT, delimiter TEXT, import_meta_json TEXT,
         created_at TEXT)
sheet_connections(id TEXT PK, project_id TEXT, spreadsheet_id TEXT, gid TEXT NULL,
         sheet_name TEXT NULL, public_url TEXT, last_status TEXT, last_checked_at TEXT,
         last_error TEXT NULL, last_version_id TEXT NULL)
columns(id TEXT PK, version_id TEXT, name TEXT, inferred_role TEXT,
         user_role TEXT NULL, n_missing INTEGER, n_unique INTEGER, is_constant INTEGER,
         sample_preview TEXT)
experiments(id TEXT PK, project_id TEXT, dataset_id TEXT, created_at TEXT)
experiment_revisions(id TEXT PK, experiment_id TEXT, revision INTEGER, config_json TEXT,
         created_at TEXT)
jobs(id TEXT PK, project_id TEXT, experiment_revision_id TEXT NULL, model_id TEXT NULL,
         type TEXT, /* train|predict */ status TEXT, /* queued|running|succeeded|failed|canceled */
         attempt_id TEXT, lease_until TEXT NULL, heartbeat_at TEXT NULL,
         progress_json TEXT, error TEXT NULL, created_at TEXT, started_at TEXT, finished_at TEXT)
models(id TEXT PK, project_id TEXT, job_id TEXT, dataset_version_id TEXT,
         task TEXT, artifact_path TEXT, metrics_json TEXT, created_at TEXT)
reports(id TEXT PK, job_id TEXT, report_json TEXT, plot_dir TEXT)
predictions(id TEXT PK, job_id TEXT, model_id TEXT, input_version_id TEXT,
         output_path TEXT, created_at TEXT)
conversations(id TEXT PK, project_id TEXT, created_at TEXT)
messages(id TEXT PK, conversation_id TEXT, role TEXT, content TEXT,
         client_id TEXT UNIQUE, status TEXT, /* complete|streaming|interrupted|failed */
         provider TEXT NULL, model TEXT NULL, usage_json TEXT NULL, created_at TEXT)
sample_disclosure(dataset_id TEXT PK, enabled INTEGER, previewed_at TEXT NULL)
```

Create a default project named `Local workspace` on first boot.

## REST (prefix `/api`)

All mutating routes are CSRF-protected for cookie sessions (SameSite=Lax + origin check). JSON APIs from the local UI send `X-Requested-With: mcp-cdp`.

- `GET /api/health` → `{ok:true}`
- `GET /api/status` → `{llm_configured:bool, provider:str|null, mcp_token_set:bool}` (never keys)
- `GET/POST /api/projects`
- `POST /api/projects/{id}/datasets/upload` multipart `file`
- `POST /api/projects/{id}/datasets/sheets` `{url, gid?, header_row?, name?}`
- `POST /api/projects/{id}/sheet-connections/{id}/refresh`
- `GET /api/projects/{id}/datasets`
- `GET /api/projects/{id}/datasets/{id}` including current version + profile summary
- `GET /api/projects/{id}/datasets/{id}/preview?n=20`
- `POST /api/projects/{id}/datasets/{id}/disclosure/preview` records `previewed_at`; returns a bounded sample (≤5 rows). Does not enable sharing.
- `PUT /api/projects/{id}/datasets/{id}/disclosure` `{enabled:bool}` — enable requires a prior preview; disable applies immediately
- `PUT /api/projects/{id}/datasets/{id}/columns` `{columns:[{name, role}]}`
- `POST /api/projects/{id}/experiments` `{dataset_id, config}`
- `POST /api/projects/{id}/experiments/{id}/submit` → `{job_id, status:"queued"}` immediately
- `GET /api/projects/{id}/jobs/{id}`
- `GET /api/projects/{id}/jobs/{id}/events` SSE `{type, ...}`
- `POST /api/projects/{id}/jobs/{id}/cancel`
- `GET /api/projects/{id}/reports/{job_id}`
- `POST /api/projects/{id}/predict` multipart or `{model_id, dataset_id}` → `{job_id}`
- `GET /api/projects/{id}/predictions/{id}/download`
- Chat: `POST /api/conversations`, `GET /api/conversations/{id}/messages`,
  `POST /api/conversations/{id}/messages` body `{content, client_id}` → SSE

Limits enforced server-side: 50 MB, 100_000 rows, 100 forecast series, horizon 1–90.

## Experiment config JSON

```json
{
  "task": "classification|regression|forecast",
  "target": "col",
  "features": ["..."],
  "excluded": ["..."],
  "roles": {"col": "numeric|categorical|text|date|identifier|excluded"},
  "group_columns": ["sku", "store"],
  "date_column": "date",
  "frequency": "D|W|M",
  "horizon": 14,
  "duplicate_timestamp": "reject|aggregate_mean",
  "entity_column": null,
  "split": "random|group|time",
  "test_size": 0.2,
  "budget": "quick|thorough",
  "seed": 42
}
```

## ML interface (platform worker → ML)

`app/ml/runner.py` must export:

```python
def run_train(job: dict, paths: dict) -> dict: ...
def run_predict(job: dict, paths: dict) -> dict: ...
```

`job` contains `id`, `type`, `config` (experiment config), `dataset_normalized_path`, optional `predict_path`.
`paths` contains `artifact_dir`.
Return dict includes `metrics`, `model_path`, `plot_files`, `warnings`, `selected_candidate`.
ML must not import `app.api`, `app.mcp`, `app.services.chat`, `openai`, or `anthropic`.

Candidates: dummy, regularized linear, XGBoost. Select on validation, report test separately.
Forecast: last-value + seasonal-naive + pooled XGBoost when applicable.

## MCP

Mount at `/mcp` with `streamable_http_path="/"`. Lifespan MUST run `mcp.session_manager.run()` after `streamable_http_app()` is created. Bearer token from env `MCP_TOKEN`. Bind loopback.

Tools (all project-scoped except `list_projects`):
`list_projects`, `get_project`, `list_datasets`, `inspect_dataset`, `get_profile`, `get_report`,
`connect_google_sheet`, `refresh_google_sheet`, `configure_experiment`, `submit_training`,
`get_job`, `cancel_job`, `submit_predict`, `import_local_file` (stdio only).

Writes require `confirm: true`. Long jobs return `{job_id}` immediately.
Stdio bridge: `python -m app.mcp.stdio_bridge` forwards to `http://127.0.0.1:8765/mcp`. Logs to stderr only.

## Chat

User row persisted before generation. Assistant row `streaming` then `complete`/`interrupted`/`failed`.
Idempotent on `client_id`. Default context: schema, aggregates, experiment settings, computed metrics.
Raw samples only if `sample_disclosure.enabled` after `POST .../disclosure/preview`. Enable/disable is `PUT .../disclosure`. Max 6 tool calls per turn. Tools call the same services as REST. There is no MCP write tool for disclosure.
If no API key, chat explains that modeling still works from the cards.

## Google Sheets

Never fetch the user URL. Parse id + gid, then GET
`https://docs.google.com/spreadsheets/d/{id}/export?format=csv&gid={gid}`.
Allowlist `docs.google.com`, `spreadsheets.google.com`, `*.googleusercontent.com`.
Login wall / 401 / 403 → `private`. 404 → `revoked`. Hash canonical CSV bytes. Same hash → no new version.

## Frontend

Vite + React + TypeScript in `frontend/`. Dev server on **5173**. Proxy `/api` to `http://127.0.0.1:8765`.
Chat is the center. Guided cards: Data → What to predict → Columns → Confirm & train → How good is it? → Predict.
Non-technical copy. Drag-and-drop CSV and file picker. Sheets URL field. Honest empty/error states.
Do not require an LLM key for the modeling path.

## Tests each owner must add

- Platform: CSV limits, bad CSV rejected, identifier `00123` preserved, sheets URL parser + mocked fetch, job enqueue/cancel, origin check.
- ML: leakage (preprocessor fitted on train only), classification/regression metric recomputation, forecast group isolation, import linter that `app.ml` does not import web/LLM.
- Chat/MCP: client_id idempotency, no keys in responses, tool confirm required, stdio path traversal rejected.
- Frontend: not unit-tested in this split; keep components typed and runnable.

## App factory hook

`app.main:create_app()` must:

1. init db
2. include platform routers
3. include chat router if present
4. create MCP ASGI app then enter `session_manager.run()` in lifespan
5. start the one worker (max 1 running ML job, queue 10)
6. optionally mount `frontend/dist` at `/`
