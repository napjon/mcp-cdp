# mcp-cdp design

Normative product design for the local AutoML workspace. Implementation evidence is tracked in [eval-questions.md](eval-questions.md); a written design does not count as a Pass.

[CONTRACT.md](CONTRACT.md) is frozen. Do not rewrite it. If these documents and the contract ever disagree, the contract wins on schema, routes, ports, limits, MCP tools, and owner split.

## Documents

| Document | What it covers |
|---|---|
| [CONTRACT.md](CONTRACT.md) | Frozen implementation contract: schema, REST, MCP, ML interface, limits |
| [architecture.md](architecture.md) | Process, SQLite, worker, REST, MCP, frontend, data flow |
| [decisions.md](decisions.md) | ADRs for locked product choices |
| [data-and-ml.md](data-and-ml.md) | Ingest, profiling, roles, train/predict pipelines, leakage, metrics, artifacts |
| [chat-and-mcp.md](chat-and-mcp.md) | Chat protocol, sample disclosure, tool policy, Codex / Claude Code / Claude Desktop |
| [fieldbook.md](fieldbook.md) | Product fieldbook (original 44-item artifact is missing; not reconstructed) |
| [eval-questions.md](eval-questions.md) | 100-question release checklist with Pass / Fail / Not verified protocol |

## Locked product choices

Recorded as ADRs in [decisions.md](decisions.md):

- Local workspace, SQLite, one user
- MCP clients control this app
- Chat via API keys (OpenAI / Anthropic), server-side
- Chat + guided cards
- Forecast: products and locations (grouped)
- Up to 100,000 rows
- LLM gets summaries only by default
- Quick comparison AutoML default
- Self-contained (no sibling repos)
- Google Sheets public-link import, manual refresh

## Runtime constants (from the contract)

| Item | Value |
|---|---|
| Bind | `127.0.0.1:8765` only |
| Frontend dev | Vite on **5173**, proxy `/api` to the API |
| Upload / row limits | 50 MB, 100,000 rows |
| Forecast | 100 series, horizon 1–90 |
| SQLite | `data/mcp_cdp.sqlite` (WAL, foreign keys on) |
| Default project | `Local workspace` |

## Owner split

Design owns `design/**` except `CONTRACT.md`. Platform, ML, frontend, and Chat+MCP owners implement against the contract and these documents. Do not invent a second schema or a second API.
