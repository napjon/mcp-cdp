# Product fieldbook

## Missing artifact

The original **44-item** Claude fieldbook artifact was **not recoverable** in this session. This file does **not** reconstruct numbered items, titles, or coverage claims from that artifact.

Eval question 6 asks whether the fieldbook accounts for all 44 framework items with decisions, evidence, or explicit non-applicability. Until the original list is recovered, or product explicitly retires that gate, question 6 stays **Not verified**. Inventing a substitute 44-item list would make the register look complete while being false.

What this fieldbook does record: goal, users, locked decisions (from [decisions.md](decisions.md)), alternatives, revisit triggers, and verification gates mapped to the **100** release questions in [eval-questions.md](eval-questions.md).

## Goal

A local AutoML workspace an operator can run on one machine: import a CSV or a public Google Sheet, say what to predict, train real classification / regression / grouped SKU+store models, read an honest report, and export predictions. Chat is available when an OpenAI or Anthropic key is configured. Codex, Claude Code, and Claude Desktop control the same workspace through MCP. The core path works with no LLM key.

This is the product, not a stand-in for `data-platform`. That product's predictions are inference heuristics; this product trains and stores fitted pipelines.

## Users

- **Local operator** — one person with a table (often product × location demand, or a labeled CSV). Uses the browser on loopback. Copy on the cards is non-technical.
- **Agent operator** — the same person, or a coding agent, driving the workspace from Codex, Claude Code, or Claude Desktop. Writes are confirmed. Long jobs return a job id.

There is no second tenant, no shared cloud account, and no sibling-repo install.

## Locked decisions

See [decisions.md](decisions.md) for alternatives, tradeoffs, owner (**product**), review date (**2026-09-13**), and change conditions. Summary:

| ADR | Lock |
|---|---|
| 001 | Local workspace, SQLite, one user, bind `127.0.0.1:8765` |
| 002 | MCP clients control this app |
| 003 | Chat keys server-side (OpenAI / Anthropic) |
| 004 | Chat + guided cards; train without a key |
| 005 | Forecast grouped by product and location |
| 006 | 100,000 rows, 50 MB, 100 series, horizon 1–90 |
| 007 | LLM summaries only by default |
| 008 | Quick AutoML default (dummy + linear + XGBoost) |
| 009 | Self-contained; no sibling repos |
| 010 | Public Google Sheets link, manual refresh |

Contract constants that travel with these locks: default project name `Local workspace`; WAL SQLite at `data/mcp_cdp.sqlite`; one ML worker and queue 10; CSRF on cookie mutations; MCP bearer `MCP_TOKEN`; Sheets export URL constructed from parsed id/gid, never the user URL.

## Alternatives considered

Captured per ADR. Cross-cutting rejects:

- **Hosted multi-user + Postgres** — wrong operator and threat model.
- **UI-only or REST-only agents** — fails the requirement that MCP clients control the app.
- **Browser-held provider keys** — leaks into DevTools and message storage.
- **Chat-only or cards-only** — blocks either non-technical completion or agent control.
- **Single-series forecast or deep sequence defaults** — misses SKU+store tables on a laptop budget.
- **Unbounded table size / silent sampling** — hides failure and blows the one-worker budget.
- **Always-on row samples to the LLM** — unnecessary disclosure.
- **Thorough or third-party AutoML as the default engine** — slow, harder to leakage-test, extra families without a dummy.
- **Importing `data-platform` prediction APIs** — heuristic inference, not fitted pipelines.
- **Google OAuth / private sheets / polling** — credentials and surprise network fetch this product will not take.

## Revisit triggers

Re-open the matching ADR if any of these become true:

- A second concurrent user or a non-loopback bind is required (ADR-001).
- Target MCP clients drop the protocol, or confirm-on-write cannot stop destructive submits (ADR-002).
- A local model meets the forty-case assistant eval (Q98) without cloud keys (ADR-003).
- Usability (Q100) shows operators never use cards or never use chat (ADR-004).
- Hierarchy or >100 series is a measured requirement (ADR-005).
- The 10k-row Quick benchmark (Q99) fails, or median tables exceed 100k rows (ADR-006).
- Operators cannot finish configuration without samples, with evidence (ADR-007).
- Quick systematically loses to dummy on leakage-safe evals (ADR-008).
- Product is merged into the other platform **and** that platform adopts this training pipeline (ADR-009).
- Private sheets become a stated requirement, or the CSV export path dies (ADR-010).

## Verification gates (mapped to the 100 questions)

A gate passes only when every listed question is **Pass** with evidence in [eval-questions.md](eval-questions.md). Design prose is not evidence. Current status of all questions: **Not verified**.

| Gate | Protects | Questions |
|---|---|---|
| A. Completeness and architecture | Goal; ADR-001, 004, 009; owner split | 1–10 |
| B. CSV ingest | ADR-006; identifier preservation; versions | 11–20 |
| C. Google Sheets | ADR-010 | 21–30 |
| D. Problem definition and quality | Roles; confirm-before-train; immutable revisions | 31–40 |
| E. Classification / regression | ADR-008; leakage; metrics | 41–50 |
| F. Grouped forecast | ADR-005; 100 series; horizon 1–90 | 51–60 |
| G. Reports and reuse | Honest metrics; fitted pipeline reuse | 61–70 |
| H. Chat and providers | ADR-003, 007; six-tool cap | 71–80 |
| I. MCP clients | ADR-002; Codex / Claude Code / Claude Desktop | 81–90 |
| J. Reliability and release evidence | Loopback, worker, adversarial tests, Q98–100 | 91–100 |

Question **6** is inside Gate A and is blocked on recovery of the 44-item artifact (or an explicit product decision to retire that question). Do not mark Gate A complete while Q6 is open.

Question **5** (design docs present and consistent with implementation) needs both this `design/` tree and an implementation diff; the documents alone are not a Pass.

Human gates: Q98 (forty-case assistant eval, three runs, 95% workflow/tool correctness, zero fabricated metrics or prohibited disclosures), Q99 (10k-row Quick ≤ 5 min, control API p95 < 500 ms), Q100 (four of five non-technical operators finish import-to-predict in ten minutes and can name the model's main limitation).
