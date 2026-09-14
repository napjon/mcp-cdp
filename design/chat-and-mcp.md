# Chat and MCP

Chat is an optional assistant over the same services as the guided cards. MCP is a required control surface for Codex, Claude Code, and Claude Desktop. Neither path is allowed to grow a second business-rule stack.

## Chat protocol

### HTTP

| Method | Path | Body / result |
|---|---|---|
| POST | `/api/conversations` | Creates a conversation on a project |
| GET | `/api/conversations/{id}/messages` | Persisted history |
| POST | `/api/conversations/{id}/messages` | `{content, client_id}` → SSE |

`GET /api/status` reports `{llm_configured, provider, mcp_token_set}` and never keys.

### Persistence and idempotency

1. The **user** row is written before any provider call.
2. An **assistant** row is created with `status=streaming`, then moved to `complete`, `interrupted`, or `failed`.
3. `messages.client_id` is unique. A retry with the same `client_id` does not insert a second user row and does not start a second generation if the first finished.
4. Reconnect reads SQLite. It must not duplicate messages or silently re-run a completed tool/train action.
5. `provider`, `model`, and `usage_json` are stored on the assistant row when known.

### SSE

The POST-messages response is a stream of events the UI can render incrementally. Interrupted streams leave the assistant row `interrupted` with whatever text was persisted. Provider failures leave `failed` with a safe error string (no key material).

### When no key is configured

Chat still accepts the route. The assistant (or a deterministic local message) explains that **modeling continues from the cards**. Import, train, report, and predict do not require OpenAI or Anthropic.

### Providers

OpenAI and Anthropic are both supported. Authentication, streaming, failure reporting, and recording of provider + model are required. Keys exist only in the server environment. They are excluded from:

- browser storage
- SQLite
- log lines
- error responses
- `/api/status`

Context and output limits are enforced. Cost figures, if shown, are **estimates** derived from `usage_json`, labeled as such.

## Default context vs sample disclosure

### Default (disclosure off)

The provider may receive:

- column schema and roles
- aggregates and profiler summaries (counts, missing, unique, constant flags)
- experiment settings (the frozen config)
- computed metrics from stored reports

The provider may **not** receive raw row samples, original CSV bytes, or `sample_preview` contents.

### Opt-in

Table `sample_disclosure(dataset_id PK, enabled, previewed_at)`:

1. Operator enables sharing **per dataset**.
2. UI shows a preview of what would be sent and records `previewed_at`.
3. Only after that preview may subsequent chat turns include a bounded sample.
4. Setting `enabled=0` applies to later requests immediately.
5. MCP tools use the same flag. There is no MCP-only bypass.

Eval questions 75–76, 79, and 90 rest on this policy. Metric claims in chat must cite persisted reports; missing evidence is stated as missing, never invented.

### REST (sample disclosure)

[CONTRACT.md](CONTRACT.md) locks the `sample_disclosure` table, the summaries-only default, and these routes:

| Method | Path | Body / result |
|---|---|---|
| POST | `/api/projects/{project_id}/datasets/{dataset_id}/disclosure/preview` | Records `previewed_at`. Returns a bounded sample (≤5 rows) for the Data card. Does not set `enabled=1`. |
| PUT | `/api/projects/{project_id}/datasets/{dataset_id}/disclosure` | `{enabled: bool}`. Enable is rejected until a preview has been recorded. Disable applies to later chat and MCP requests immediately. |

The Data card calls these routes. Chat and MCP tools read the same `sample_disclosure` row. MCP has no write tool for disclosure; clients stay summaries-only unless the operator already enabled sharing in the UI. Do not invent a parallel table, a second policy, or a different path (for example `.../sample-disclosure`).

## Tool policy (in-app chat)

Chat tools call the **same services** as REST. They are not a second implementation.

| Rule | Detail |
|---|---|
| Schema | Arguments validated against the tool schema |
| Scope | Active project only |
| Budget | **Max 6 tool calls per turn** |
| Training | Assistant may propose config; training starts only when the operator acts on Confirm & train, or when MCP `submit_training` is called with `confirm: true` |
| Writes | Same validations as REST (limits, roles, splits) |
| Forbidden | Shell, SQL, code execution, unrestricted URL fetch |

## MCP surface

Mount: `/mcp` with `streamable_http_path="/"`. Lifespan **must** enter `mcp.session_manager.run()` after the Streamable HTTP app is created. Bind: loopback. Auth: `Authorization: Bearer` from `MCP_TOKEN`.

The stdio bridge (`python -m app.mcp.stdio_bridge`, script `mcp-cdp-stdio`) forwards to `http://127.0.0.1:8765/mcp`. It must not spawn a worker or open a second SQLite file. Stdout is protocol only; logs go to stderr.

### Tools

All project-scoped except `list_projects`:

| Tool | Access | Notes |
|---|---|---|
| `list_projects` | read | |
| `get_project` | read | |
| `list_datasets` | read | |
| `inspect_dataset` | read | |
| `get_profile` | read | Summaries, not raw rows unless disclosure allows a bounded preview already shown in REST |
| `get_report` | read | Persisted report only |
| `connect_google_sheet` | write | Same public-link rules as REST |
| `refresh_google_sheet` | write | Manual; hash-equals skips a new version |
| `configure_experiment` | write | Same config JSON as REST |
| `submit_training` | write | Returns `{job_id}` immediately |
| `get_job` | read | |
| `cancel_job` | write | |
| `submit_predict` | write | Returns `{job_id}` immediately |
| `import_local_file` | write | **Stdio only**; configured directories; reject traversal and symlink escapes |

Writes require `confirm: true`. Tool names, descriptions, input/output schemas, and read/write annotations must match this table so clients can discover them.

Long operations return a job id and a bounded summary plus artifact references. They do not block the MCP session on XGBoost fit.

Equivalent REST and MCP calls must produce equivalent configs, validation errors, and artifact bytes.

## Connecting clients

The HTTP app must already be running on `127.0.0.1:8765`. MCP does not start a second copy of the product.

Set `MCP_TOKEN` in the app environment and in the client configuration to the same value.

### Codex

`~/.codex/config.toml` or a trusted project `.codex/config.toml`:

```toml
[mcp_servers.mcp-cdp]
url = "http://127.0.0.1:8765/mcp"
bearer_token_env_var = "MCP_TOKEN"
```

Stdio alternative (still requires the HTTP app up, because the bridge forwards):

```toml
[mcp_servers.mcp-cdp]
command = "mcp-cdp-stdio"
```

Use Streamable HTTP when the client supports it. Confirm with Codex `/mcp`.

### Claude Code

CLI:

```bash
claude mcp add --transport http mcp-cdp http://127.0.0.1:8765/mcp \
  --header "Authorization: Bearer ${MCP_TOKEN}"
```

JSON (`.mcp.json` or user config). `type` is required for URL entries; `streamable-http` is accepted as an alias of `http`:

```json
{
  "mcpServers": {
    "mcp-cdp": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp",
      "headers": {
        "Authorization": "Bearer ${MCP_TOKEN}"
      }
    }
  }
}
```

### Claude Desktop

Loopback Streamable HTTP is not the reliable Desktop path (Custom Connectors expect a reachable remote server). Use the **stdio bridge** in `claude_desktop_config.json`:

macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mcp-cdp": {
      "command": "mcp-cdp-stdio",
      "env": {
        "MCP_TOKEN": "<same token as the running app>"
      }
    }
  }
}
```

`mcp-cdp-stdio` must be on `PATH` (install the package) or `command`/`args` must invoke `python -m app.mcp.stdio_bridge` with `cwd` set to the checkout. The HTTP app remains the only worker.

### Shared client rules

- No client should launch a second `mcp-cdp` HTTP process “to be helpful.”
- `import_local_file` exists only on stdio. HTTP MCP clients use upload/Sheets tools instead.
- If `MCP_TOKEN` is unset on the server, `/api/status` shows `mcp_token_set: false` and MCP auth fails closed.
