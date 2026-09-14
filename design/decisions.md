# Architectural decisions

Locked product choices. Each record lists alternatives, tradeoffs, owner, review date, and conditions that would change it. Implementation Pass/Fail is not claimed here; see [eval-questions.md](eval-questions.md).

Owner for every decision in this file: **product**.  
Review date for every decision in this file: **2026-09-13**.

---

## ADR-001 — Local workspace, SQLite, one user

**Status:** Locked

**Decision:** The product is a single-user app on the operator's machine. State lives in SQLite at `data/mcp_cdp.sqlite` (WAL, foreign keys on). The HTTP server binds `127.0.0.1:8765` only. First boot creates a project named `Local workspace`.

**Context:** The operator needs to import a table, train, and export predictions without standing up Postgres, object storage, or an identity provider.

**Alternatives:**

- Hosted multi-user service with Postgres and accounts.
- Embedded Postgres or DuckDB.
- Browser-only (no durable process).

**Tradeoffs:** SQLite plus loopback keeps backup to a directory copy and removes remote attack surface. It also means no concurrent operators, no LAN bind, and no team sharing. WAL and foreign keys are mandatory so jobs and versions stay consistent under one writer.

**Conditions that would change this:** A required second concurrent user; a need to bind beyond loopback; SQLite write contention that breaks the one-worker design after evidence on the reference laptop.

---

## ADR-002 — MCP clients control this app

**Status:** Locked

**Decision:** Codex, Claude Code, and Claude Desktop drive the same domain services as the browser. MCP is mounted at `/mcp` (Streamable HTTP). Writes require `confirm: true`. Long jobs return `{job_id}` immediately. A stdio bridge (`python -m app.mcp.stdio_bridge` / `mcp-cdp-stdio`) forwards to `http://127.0.0.1:8765/mcp` and does not start a second worker or database.

**Context:** The product is meant to be operated by coding agents as well as by a person at the guided cards.

**Alternatives:**

- UI-only product.
- Public unauthenticated REST as the agent surface.
- MCP read-only (inspect, no submit).

**Tradeoffs:** Agents can import, train, and predict, which is the point. That requires a bearer token (`MCP_TOKEN`), project scoping, confirm-on-write, and no tools for shell, SQL, arbitrary code, or unrestricted HTTP. Stdio exists because Claude Desktop is reliable on stdio for a loopback server; Streamable HTTP is native for Codex and Claude Code.

**Conditions that would change this:** Target clients drop MCP; a confirmed class of accidental destructive submits that confirm flags cannot stop; a requirement that the app run without any agent surface.

---

## ADR-003 — Chat via OpenAI / Anthropic keys, server-side

**Status:** Locked

**Decision:** Optional chat uses OpenAI or Anthropic. Keys stay in the server environment. The browser, SQLite, logs, and error bodies never receive keys. `GET /api/status` returns `{llm_configured, provider, mcp_token_set}` only.

**Context:** Chat is useful for explaining reports and proposing configuration. It is not required to train.

**Alternatives:**

- Keys typed into the browser and sent per request.
- OAuth device flow to a hosted proxy.
- No chat product at all.
- Local-only models.

**Tradeoffs:** Server-side keys avoid leaking credentials into DevTools and message rows. The operator must configure environment variables. Chat is unavailable without a key; the cards still run. Cost figures, when shown, are labeled estimates from recorded `usage_json`, not invoices.

**Conditions that would change this:** A local model that meets the forty-case assistant eval (question 98) without a cloud key; a secret-store UI that still never writes keys to SQLite or the client; evidence that browser-held keys are the only usable setup for the audience.

---

## ADR-004 — Chat plus guided cards

**Status:** Locked

**Decision:** The UI is chat-centered with a fixed card sequence: Data → What to predict → Columns → Confirm & train → How good is it? → Predict. The core import–train–predict path works with no LLM key. Chat may propose configuration; it must not start training until the user acts on the review card (or an MCP client sends `submit_training` with `confirm: true`).

**Context:** Non-technical operators need visible structure. Agent operators need chat and MCP. Neither path is optional in the product, but LLM failure must not block modeling.

**Alternatives:**

- Chat-only (no cards).
- Wizard-only (no chat).
- Notebook / script UI.

**Tradeoffs:** Two surfaces must stay consistent (same services, same validation). Cards keep the source of truth for what will train. Chat without keys still explains that modeling continues from the cards.

**Conditions that would change this:** Usability evidence (question 100) that operators never use one of the two surfaces; a decision to make MCP the only non-browser control and drop in-app chat.

---

## ADR-005 — Forecast products and locations (grouped)

**Status:** Locked

**Decision:** Forecasting is grouped time series for products and locations (SKU + store). Config carries `group_columns`, `date_column`, `frequency` `D|W|M`, `horizon` 1–90, and `duplicate_timestamp` `reject|aggregate_mean`. Candidates: last-value, seasonal-naive, and pooled XGBoost when applicable. Cap: 100 series.

**Context:** The intended forecast job is demand-style series keyed by product and site, not a single global trend line and not hierarchical reconciliation.

**Alternatives:**

- Single-series forecast only.
- Hierarchical / reconciled forecast.
- Deep models (sequence nets) as the default.

**Tradeoffs:** Group isolation and pooled XGBoost match SKU×store tables on a laptop. The 100-series and 90-step caps reject larger planning models. No uncalibrated uncertainty bands are presented as reliable. Forecast holdout is the configured horizon, not the classification/regression `test_size` of 0.2.

**Conditions that would change this:** A required hierarchy (SKU → category → chain); routine datasets above 100 series with measured need; evidence that last-value / seasonal-naive / pooled XGBoost cannot beat a simpler single-series baseline on the product's own eval sets.

---

## ADR-006 — Up to 100,000 rows

**Status:** Locked

**Decision:** Server-side rejection above **100,000** rows and **50 MB** per import. Forecast adds **100 series** and **horizon 1–90**. These limits apply equally to UI, REST, and MCP.

**Context:** Training runs on the operator laptop next to the API. The release bar (question 99) is a 10,000-row Quick job in five minutes with control API p95 under 500 ms during training.

**Alternatives:**

- No cap (until memory fails).
- Automatic sampling to a working subset.
- Out-of-core / distributed training.

**Tradeoffs:** Caps make failure modes explicit and keep the one-worker process schedulable. They refuse legitimate larger tables. Sampling is not silent; oversize input is an error.

**Conditions that would change this:** Measured failure of question 99 at 10k rows; a documented audience whose median table exceeds 100k rows; a worker redesign that isolates ML in a way that safely raises the cap.

---

## ADR-007 — LLM gets summaries only by default

**Status:** Locked

**Decision:** Default provider context is schema, aggregates, experiment settings, and computed metrics. Raw row samples are sent only when `sample_disclosure.enabled` is set for that dataset after an explicit preview (`previewed_at`). Revocation applies to subsequent requests. MCP tools obey the same policy.

**Context:** Tables can contain identifiers and customer fields. Chat should configure and explain models without shipping the table to a provider.

**Alternatives:**

- Always attach a sample window.
- Opt-out instead of opt-in.
- Retrieval over raw rows.

**Tradeoffs:** Summaries-first reduces accidental disclosure and still allows column/role help. Some data-quality advice is weaker without samples. The preview step is extra friction by design.

**Conditions that would change this:** Evidence that operators cannot finish configuration without samples (question 100) *and* a smaller, safer sample contract; a local model with no network egress.

---

## ADR-008 — Quick comparison AutoML as default

**Status:** Locked

**Decision:** Default budget is **Quick**: dummy baseline + regularized linear + XGBoost (forecast: last-value + seasonal-naive + pooled XGBoost when applicable). Selection uses validation only; test is reported separately. **Thorough** is the same families with a larger search; it must still finish and must not publish an incomplete model. Seed default is 42.

**Context:** Operators need a real comparison, including a dummy, without waiting on an unbounded search.

**Alternatives:**

- Single model (for example XGBoost only).
- External AutoML stacks as the engine.
- Thorough as the default budget.

**Tradeoffs:** Three families keep the pipeline understandable and leakage-testable. They will lose to specialist models on some tables. Dummy is required so “better than nothing” is visible even when linear/XGBoost are weak.

**Conditions that would change this:** Quick systematically loses to dummy on reference tasks after leakage-safe eval; Thorough cannot finish within laptop budgets; a new family is added only with the same validation/test split rules.

---

## ADR-009 — Self-contained (no sibling repos)

**Status:** Locked

**Decision:** This repository is the whole product. No imports from `data-platform`, cube-8s, or other sibling apps. No shared prediction service.

**Context:** The other platform's prediction path is heuristic / LLM inference, not a fitted local pipeline. Folding into it would ship the wrong method.

**Alternatives:**

- Call `data-platform` prediction APIs.
- Extract a shared ML wheel from both repos.
- Run as a module inside the multi-tenant suite.

**Tradeoffs:** Duplication of generic HTTP/auth ideas is accepted. Independence preserves leakage-safe training, SQLite, and MCP control. It also means this app will not gain that suite's connectors, RBAC, or dashboards.

**Conditions that would change this:** Product is explicitly merged into the other platform **and** that platform replaces heuristics with this training pipeline; or a extracted library that still trains for real and does not pull web/LLM into `app.ml`.

---

## ADR-010 — Google Sheets public-link import, manual refresh

**Status:** Locked

**Decision:** Import Sheets that are shared **Anyone with the link**. No Google sign-in, no OAuth client, no Drive API. Never fetch the user-supplied URL. Parse spreadsheet id and optional gid, then GET the CSV export on an allowlist. Refresh is a user/MCP action, not a poller. Unchanged canonical bytes (same hash) do not create a version.

**Context:** Operators often live in a spreadsheet. Private-sheet access would force Google app credentials and a security review this product is not taking.

**Alternatives:**

- OAuth and private files.
- Periodic polling.
- Browser extension that reads the sheet locally.

**Tradeoffs:** Public-link-only is honest: private / 401 / 403 → `private`; 404 → `revoked`. Manual refresh avoids surprise fetches. Operators with private finance sheets cannot import them here.

**Conditions that would change this:** Private sheets become a stated requirement; Google disables the CSV export path; allowlist bypasses appear in adversarial tests (question 96) that cannot be fixed without dropping Sheets.
