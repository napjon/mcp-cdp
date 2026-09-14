import type {
  AppStatus,
  ChatMessage,
  ColumnProfile,
  ColumnRole,
  Conversation,
  Dataset,
  DatasetDetail,
  Experiment,
  ExperimentConfig,
  Job,
  JobProgress,
  PreviewTable,
  Project,
  ReportPayload,
  SampleDisclosure,
  SubmitJobResponse,
} from './types'

export const REQUESTED_WITH = 'mcp-cdp'

export class ApiError extends Error {
  status: number
  body: unknown

  constructor(message: string, status = 0, body: unknown = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function pickString(value: unknown, keys: string[]): string | null {
  if (!isRecord(value)) return null
  for (const key of keys) {
    const item = value[key]
    if (typeof item === 'string' && item.trim()) return item
  }
  return null
}

function asArray<T>(value: unknown, keys: string[]): T[] {
  if (Array.isArray(value)) return value as T[]
  if (!isRecord(value)) return []
  for (const key of keys) {
    const item = value[key]
    if (Array.isArray(item)) return item as T[]
  }
  return []
}

function asObject<T>(value: unknown, keys: string[]): T | null {
  if (!isRecord(value)) return null
  for (const key of keys) {
    const item = value[key]
    if (isRecord(item)) return item as T
  }
  return value as T
}

export const SHEET_TAB_HELP =
  'This link doesn’t say which tab to use. Open the tab you want in Google Sheets, copy the address from the browser — it should include #gid= — and paste that here.'

export function sheetTabGid(url: string): string | null {
  const trimmed = url.trim()
  if (!trimmed) return null
  try {
    const parsed = new URL(trimmed)
    const queryGid = parsed.searchParams.get('gid')
    if (queryGid != null && /^\d+$/.test(queryGid)) return queryGid
    const hashGid = /(?:^|[?#&])gid=(\d+)/.exec(parsed.hash)
    if (hashGid) return hashGid[1]
  } catch {
    /* not a full URL; still accept a gid fragment */
  }
  const match = /[#?&]gid=(\d+)/.exec(trimmed)
  return match ? match[1] : null
}

export function isGidUnspecified(code: string, text = ''): boolean {
  const blob = `${code} ${text}`.toLowerCase()
  if (
    code === 'gid_unspecified' ||
    code === 'missing_gid' ||
    code === 'gid_required' ||
    code === 'no_gid' ||
    code === 'tab_unspecified'
  ) {
    return true
  }
  return /(?:gid|tab).*(?:unspecified|missing|required)|(?:unspecified|missing).*(?:gid|tab)/.test(blob)
}

export function plainError(err: unknown): string {
  if (err instanceof ApiError) return err.message
  if (err instanceof DOMException && err.name === 'AbortError') {
    return 'Stopped.'
  }
  if (err instanceof TypeError) {
    return 'Can’t reach the local app. Make sure it is running on this computer.'
  }
  if (err instanceof Error && err.message) return err.message
  return 'Something went wrong. Try again.'
}

function extractErrorMessage(body: unknown, status: number): string {
  const code = isRecord(body) && typeof body.code === 'string' ? body.code : ''
  const detail = isRecord(body) ? body.detail ?? body.error ?? body.message ?? body.last_error : body
  const text =
    typeof detail === 'string'
      ? detail
      : Array.isArray(detail)
        ? detail
            .map((item) =>
              isRecord(item) ? String(item.msg ?? item.message ?? item.detail ?? '') : String(item),
            )
            .filter(Boolean)
            .join(' ')
        : isRecord(detail)
          ? String(detail.msg ?? detail.message ?? detail.status ?? '')
          : ''
  const lower = text.toLowerCase()

  if (code === 'too_large' || status === 413 || (lower.includes('50') && lower.includes('mb'))) {
    return 'This file is too large. Please use a spreadsheet under 50 MB.'
  }
  if (code === 'too_many_rows' || lower.includes('100_000') || lower.includes('100000') || lower.includes('too many rows')) {
    return 'This spreadsheet has too many rows. The limit is 100,000.'
  }
  if (code === 'private' || lower.includes('private')) {
    return 'This Google Sheet is private. Share it with “Anyone with the link”, then try again.'
  }
  if (code === 'revoked' || lower.includes('revoked') || (status === 404 && lower.includes('sheet'))) {
    return 'That sheet was not found. Check the link, or confirm it is still shared.'
  }
  if (code === 'not_a_sheet') {
    return 'That does not look like a Google Sheets link.'
  }
  if (isGidUnspecified(code, text)) {
    return SHEET_TAB_HELP
  }
  if (lower.includes('horizon')) {
    return 'Look-ahead must be between 1 and 90 days.'
  }
  if (lower.includes('series') && lower.includes('100')) {
    return 'There are too many separate series to forecast. The limit is 100.'
  }
  if (lower.includes('csv') || lower.includes('parse') || lower.includes('delimiter')) {
    return text.trim() || 'This file could not be read as a spreadsheet. Try exporting it as CSV.'
  }
  if (status === 404) return text.trim() || 'We couldn’t find that. It may have been removed.'
  if (status === 409) return text.trim() || 'That request was already handled.'
  if (status >= 500) return text.trim() || 'Something went wrong on the server. Try again.'
  return text.trim() || `Request failed (${status}).`
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const method = (init.method ?? 'GET').toUpperCase()
  const headers = new Headers(init.headers)
  if (method !== 'GET' && method !== 'HEAD') {
    headers.set('X-Requested-With', REQUESTED_WITH)
  }
  const isForm = typeof FormData !== 'undefined' && init.body instanceof FormData
  if (init.body && !isForm && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  let res: Response
  try {
    res = await fetch(path, { ...init, method, headers })
  } catch (err) {
    throw new ApiError(plainError(err), 0, null)
  }

  if (!res.ok) {
    let body: unknown = null
    const raw = await res.text()
    try {
      body = raw ? JSON.parse(raw) : null
    } catch {
      body = raw
    }
    throw new ApiError(extractErrorMessage(body, res.status), res.status, body)
  }

  if (res.status === 204) return undefined as T
  const contentType = res.headers.get('content-type') ?? ''
  if (contentType.includes('application/json')) {
    return (await res.json()) as T
  }
  const raw = await res.text()
  if (!raw) return undefined as T
  try {
    return JSON.parse(raw) as T
  } catch {
    return raw as T
  }
}

export function normalizePreview(raw: unknown): PreviewTable {
  const root = isRecord(raw) ? raw : {}
  const nested = isRecord(root.preview) ? root.preview : root
  let columns: string[] = []
  const header = nested.columns ?? nested.header ?? nested.headers ?? nested.fields
  if (Array.isArray(header)) {
    columns = header.map((item) => {
      if (typeof item === 'string') return item
      if (isRecord(item) && typeof item.name === 'string') return item.name
      return String(item)
    })
  }

  const data = nested.rows ?? nested.data ?? nested.records ?? nested.sample
  const rows: string[][] = []
  if (Array.isArray(data)) {
    for (const row of data) {
      if (Array.isArray(row)) {
        const cells = row.map((cell) => stringifyCell(cell))
        if (!columns.length) columns = cells.map((_, i) => `Column ${i + 1}`)
        rows.push(padRow(cells, columns.length))
      } else if (isRecord(row)) {
        if (!columns.length) columns = Object.keys(row)
        rows.push(columns.map((col) => stringifyCell(row[col])))
      }
    }
  }

  const shown = rows.length
  const total =
    num(nested.n_rows) ??
    num(nested.total) ??
    num(nested.row_count) ??
    num(nested.n_total) ??
    undefined

  return { columns, rows, shown, total }
}

export function attachPreviewTotal(
  preview: PreviewTable | null,
  dataset: DatasetDetail | null,
): PreviewTable | null {
  if (!preview) return preview
  const total =
    preview.total ??
    dataset?.current_version?.n_rows ??
    dataset?.profile?.n_rows ??
    dataset?.version?.n_rows
  return total != null ? { ...preview, total } : preview
}

function stringifyCell(value: unknown): string {
  if (value == null) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function padRow(cells: string[], width: number): string[] {
  if (cells.length >= width) return cells.slice(0, width)
  return [...cells, ...Array.from({ length: width - cells.length }, () => '')]
}

export function columnsFromDataset(
  dataset: DatasetDetail | null,
  preview: PreviewTable | null,
): ColumnProfile[] {
  const fromDataset =
    dataset?.columns ??
    dataset?.profile?.columns ??
    []
  if (fromDataset.length) return fromDataset.map(normalizeColumn)
  if (preview?.columns.length) {
    return preview.columns.map((name) => ({
      name,
      inferred_role: 'text' as const,
      user_role: null,
      is_constant: 0,
      sample_preview: preview.rows[0]?.[preview.columns.indexOf(name)] ?? '',
    }))
  }
  return []
}

function normalizeColumn(col: ColumnProfile): ColumnProfile {
  return {
    ...col,
    inferred_role: normalizeRole(col.inferred_role) ?? 'text',
    user_role: normalizeRole(col.user_role),
  }
}

export function normalizeRole(value: unknown): ColumnRole | null {
  if (typeof value !== 'string') return null
  const key = value.toLowerCase()
  if (key === 'numeric' || key === 'number' || key === 'float' || key === 'int') return 'numeric'
  if (key === 'categorical' || key === 'category' || key === 'enum') return 'categorical'
  if (key === 'text' || key === 'string') return 'text'
  if (key === 'date' || key === 'datetime' || key === 'timestamp') return 'date'
  if (key === 'identifier' || key === 'id') return 'identifier'
  if (key === 'excluded' || key === 'ignore' || key === 'drop' || key === 'constant') {
    return 'excluded'
  }
  return null
}

export function parseProgress(raw: unknown): JobProgress | null {
  if (raw == null) return null
  let value: unknown = raw
  if (typeof raw === 'string') {
    try {
      value = JSON.parse(raw)
    } catch {
      return { message: raw }
    }
  }
  if (!isRecord(value)) return null
  const percent =
    num(value.percent) ??
    num(value.progress) ??
    num(value.pct) ??
    (typeof value.fraction === 'number' ? value.fraction * 100 : undefined)
  return {
    percent,
    progress: num(value.progress),
    step: str(value.step) ?? str(value.stage),
    message: str(value.message) ?? str(value.status_text) ?? str(value.detail),
    stage: str(value.stage),
    prediction_id: str(value.prediction_id),
    model_id: str(value.model_id),
  }
}

function num(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function str(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined
}

export function unwrapReport(raw: unknown): ReportPayload {
  const envelope = isRecord(raw) ? raw : {}
  const nested = isRecord(envelope.report) ? envelope.report : envelope
  const payload = nested.report_json ?? nested
  let parsed: unknown = payload
  if (typeof payload === 'string') {
    try {
      parsed = JSON.parse(payload)
    } catch {
      parsed = {}
    }
  }
  const body = isRecord(parsed) ? ({ ...parsed } as ReportPayload) : {}
  if (!body.plots && Array.isArray(envelope.plots)) body.plots = envelope.plots as ReportPayload['plots']
  if (!body.plot_files && Array.isArray(nested.plot_files)) {
    body.plot_files = nested.plot_files as ReportPayload['plot_files']
  }
  if (!body.metrics && isRecord(nested.metrics)) {
    body.metrics = nested.metrics
  }
  const metrics = isRecord(body.metrics) ? body.metrics : null
  if (metrics && !body.task && typeof metrics.task === 'string') body.task = metrics.task
  if (metrics && !body.selected_candidate && typeof metrics.selected_candidate === 'string') {
    body.selected_candidate = metrics.selected_candidate
  }
  if (metrics && body.dataset_version == null) {
    if (typeof metrics.dataset_version === 'number' || typeof metrics.dataset_version === 'string') {
      body.dataset_version = metrics.dataset_version
    } else if (
      typeof metrics.dataset_version_id === 'number' ||
      typeof metrics.dataset_version_id === 'string'
    ) {
      body.dataset_version_id = String(metrics.dataset_version_id)
    }
  }
  const splitIdx = isRecord(metrics?.split_indices) ? metrics.split_indices : null
  if (body.n == null && Array.isArray(splitIdx?.test)) body.n = splitIdx.test.length
  if (body.n_val == null && Array.isArray(splitIdx?.val)) body.n_val = splitIdx.val.length
  const testBag = isRecord(metrics?.test) ? metrics.test : null
  if (body.n_test == null && typeof testBag?.n === 'number') body.n_test = testBag.n
  const valBag = isRecord(metrics?.validation) ? metrics.validation : null
  if (body.n_validation == null && typeof valBag?.n === 'number') body.n_validation = valBag.n
  if (body.units == null) {
    const units = str(nested.units) ?? str(envelope.units) ?? (metrics ? str(metrics.units) : undefined)
    if (units) body.units = units
  }
  if (!body.baseline_metrics) {
    if (isRecord(nested.baseline_metrics)) body.baseline_metrics = nested.baseline_metrics
    else if (isRecord(envelope.baseline_metrics)) body.baseline_metrics = envelope.baseline_metrics
  }
  return body
}

export function isNotFound(err: unknown): boolean {
  return err instanceof ApiError && err.status === 404
}

export function flattenMetrics(raw: unknown): Record<string, number | string> {
  if (!isRecord(raw)) return {}
  const preferred = isRecord(raw.test)
    ? raw.test
    : isRecord(raw.validation)
      ? raw.validation
      : raw
  const out: Record<string, number | string> = {}
  for (const [key, value] of Object.entries(preferred)) {
    if (typeof value === 'number' || typeof value === 'string') out[key] = value
  }
  if (preferred !== raw) {
    for (const [key, value] of Object.entries(raw)) {
      if ((typeof value === 'number' || typeof value === 'string') && out[key] == null) {
        out[key] = value
      }
    }
  }
  return out
}

export function plotUrl(
  projectId: string,
  jobId: string,
  plot: PlotLike,
): string | null {
  if (typeof plot === 'string') {
    if (isDirectUrl(plot)) return plot
    const file = plot.split(/[\\/]/).pop()
    if (!file) return null
    return `/api/projects/${projectId}/reports/${jobId}/plots/${encodeURIComponent(file)}`
  }
  if (plot.url && isDirectUrl(plot.url)) return plot.url
  const file = plot.filename ?? plot.path ?? plot.name ?? plot.url
  if (!file) return null
  if (isDirectUrl(file)) return file
  const base = file.split(/[\\/]/).pop()
  if (!base) return null
  return `/api/projects/${projectId}/reports/${jobId}/plots/${encodeURIComponent(base)}`
}

type PlotLike = { name?: string; url?: string; path?: string; filename?: string } | string

function isDirectUrl(value: string): boolean {
  return (
    value.startsWith('http://') ||
    value.startsWith('https://') ||
    value.startsWith('/') ||
    value.startsWith('data:')
  )
}

export function jobPredictionId(job: Job): string | null {
  if (job.prediction_id) return job.prediction_id
  if (job.result?.prediction_id) return job.result.prediction_id
  const progress = parseProgress(job.progress_json)
  return progress?.prediction_id ?? null
}

export function jobModelId(job: Job): string | null {
  if (job.model_id) return job.model_id
  if (job.result?.model_id) return job.result.model_id
  const progress = parseProgress(job.progress_json)
  return progress?.model_id ?? null
}

export async function getHealth(): Promise<{ ok: boolean }> {
  return request('/api/health')
}

export async function getStatus(): Promise<AppStatus> {
  return request('/api/status')
}

export async function listProjects(): Promise<Project[]> {
  const raw = await request<unknown>('/api/projects')
  return asArray<Project>(raw, ['projects', 'items'])
}

export async function createProject(name: string): Promise<Project> {
  const raw = await request<unknown>('/api/projects', {
    method: 'POST',
    body: JSON.stringify({ name }),
  })
  return (asObject<Project>(raw, ['project']) ?? raw) as Project
}

export async function uploadCsv(projectId: string, file: File, name?: string): Promise<DatasetDetail> {
  const body = new FormData()
  body.append('file', file)
  if (name) body.append('name', name)
  const raw = await request<unknown>(`/api/projects/${projectId}/datasets/upload`, {
    method: 'POST',
    body,
  })
  return (asObject<DatasetDetail>(raw, ['dataset']) ?? raw) as DatasetDetail
}

export async function previewSheet(
  projectId: string,
  input: { url: string; header_row?: number; gid?: string },
): Promise<PreviewTable> {
  const raw = await request<unknown>(`/api/projects/${projectId}/datasets/sheets/preview`, {
    method: 'POST',
    body: JSON.stringify({
      url: input.url,
      header_row: input.header_row,
      ...(input.gid ? { gid: input.gid } : {}),
    }),
  })
  return normalizePreview(raw)
}

export async function connectSheet(
  projectId: string,
  input: { url: string; gid?: string; header_row?: number; name?: string },
): Promise<DatasetDetail> {
  const raw = await request<unknown>(`/api/projects/${projectId}/datasets/sheets`, {
    method: 'POST',
    body: JSON.stringify(input),
  })
  return (asObject<DatasetDetail>(raw, ['dataset']) ?? raw) as DatasetDetail
}

export async function refreshSheet(projectId: string, connectionId: string): Promise<DatasetDetail | { ok: boolean }> {
  return request(`/api/projects/${projectId}/sheet-connections/${connectionId}/refresh`, {
    method: 'POST',
  })
}

export async function listDatasets(projectId: string): Promise<Dataset[]> {
  const raw = await request<unknown>(`/api/projects/${projectId}/datasets`)
  return asArray<Dataset>(raw, ['datasets', 'items'])
}

export async function getDataset(projectId: string, datasetId: string): Promise<DatasetDetail> {
  const raw = await request<unknown>(`/api/projects/${projectId}/datasets/${datasetId}`)
  return (asObject<DatasetDetail>(raw, ['dataset']) ?? raw) as DatasetDetail
}

export async function getPreview(
  projectId: string,
  datasetId: string,
  n = 20,
): Promise<PreviewTable> {
  const raw = await request<unknown>(
    `/api/projects/${projectId}/datasets/${datasetId}/preview?n=${encodeURIComponent(String(n))}`,
  )
  return normalizePreview(raw)
}

export async function previewDisclosure(
  projectId: string,
  datasetId: string,
): Promise<{ disclosure: SampleDisclosure; preview: PreviewTable }> {
  const raw = await request<unknown>(
    `/api/projects/${projectId}/datasets/${datasetId}/disclosure/preview`,
    { method: 'POST', body: JSON.stringify({}) },
  )
  const disclosure = normalizeDisclosure(raw, datasetId)
  return { disclosure, preview: normalizePreview(raw) }
}

export async function putDisclosure(
  projectId: string,
  datasetId: string,
  enabled: boolean,
): Promise<SampleDisclosure> {
  const raw = await request<unknown>(`/api/projects/${projectId}/datasets/${datasetId}/disclosure`, {
    method: 'PUT',
    body: JSON.stringify({ enabled }),
  })
  return normalizeDisclosure(raw, datasetId)
}

function normalizeDisclosure(raw: unknown, datasetId: string): SampleDisclosure {
  const root = isRecord(raw) ? raw : {}
  const nested = isRecord(root.disclosure) ? root.disclosure : root
  const enabledRaw = nested.enabled
  const enabled =
    enabledRaw === true ||
    enabledRaw === 1 ||
    enabledRaw === '1' ||
    enabledRaw === 'true'
      ? true
      : enabledRaw === false || enabledRaw === 0 || enabledRaw === '0' || enabledRaw === 'false'
        ? false
        : null
  return {
    dataset_id: str(nested.dataset_id) ?? datasetId,
    enabled,
    previewed_at: str(nested.previewed_at) ?? null,
    columns: Array.isArray(nested.columns) ? (nested.columns as string[]) : undefined,
    rows: Array.isArray(nested.rows) ? (nested.rows as SampleDisclosure['rows']) : undefined,
    sample_rows: typeof nested.sample_rows === 'number' ? nested.sample_rows : undefined,
    ok: nested.ok === true || nested.ok === undefined,
    error: str(nested.error),
  }
}

export async function putColumns(
  projectId: string,
  datasetId: string,
  columns: Array<{ name: string; role: ColumnRole }>,
): Promise<unknown> {
  return request(`/api/projects/${projectId}/datasets/${datasetId}/columns`, {
    method: 'PUT',
    body: JSON.stringify({ columns }),
  })
}

export async function createExperiment(
  projectId: string,
  input: { dataset_id: string; config: ExperimentConfig },
): Promise<Experiment> {
  const raw = await request<unknown>(`/api/projects/${projectId}/experiments`, {
    method: 'POST',
    body: JSON.stringify(input),
  })
  return (asObject<Experiment>(raw, ['experiment']) ?? raw) as Experiment
}

export async function submitExperiment(
  projectId: string,
  experimentId: string,
): Promise<SubmitJobResponse> {
  const raw = await request<unknown>(`/api/projects/${projectId}/experiments/${experimentId}/submit`, {
    method: 'POST',
  })
  const body = (asObject<SubmitJobResponse>(raw, ['job']) ?? raw) as SubmitJobResponse
  const jobId = body.job_id ?? pickString(raw, ['job_id', 'id'])
  if (!jobId) throw new ApiError('Training was accepted, but no job id came back.', 500, raw)
  return { job_id: jobId, status: body.status ?? 'queued' }
}

export async function getJob(projectId: string, jobId: string): Promise<Job> {
  const raw = await request<unknown>(`/api/projects/${projectId}/jobs/${jobId}`)
  return (asObject<Job>(raw, ['job']) ?? raw) as Job
}

export async function cancelJob(projectId: string, jobId: string): Promise<unknown> {
  return request(`/api/projects/${projectId}/jobs/${jobId}/cancel`, { method: 'POST' })
}

export async function getReport(projectId: string, jobId: string): Promise<ReportPayload> {
  const raw = await request<unknown>(`/api/projects/${projectId}/reports/${jobId}`)
  return unwrapReport(raw)
}

export async function submitPredict(
  projectId: string,
  input: { file?: File; model_id?: string; dataset_id?: string },
): Promise<SubmitJobResponse> {
  let raw: unknown
  if (input.file) {
    const body = new FormData()
    body.append('file', input.file)
    if (input.model_id) body.append('model_id', input.model_id)
    if (input.dataset_id) body.append('dataset_id', input.dataset_id)
    raw = await request<unknown>(`/api/projects/${projectId}/predict`, { method: 'POST', body })
  } else {
    raw = await request<unknown>(`/api/projects/${projectId}/predict`, {
      method: 'POST',
      body: JSON.stringify({ model_id: input.model_id, dataset_id: input.dataset_id }),
    })
  }
  const body = (asObject<SubmitJobResponse>(raw, ['job']) ?? raw) as SubmitJobResponse
  const jobId = body.job_id ?? pickString(raw, ['job_id', 'id'])
  if (!jobId) throw new ApiError('Scoring was accepted, but no job id came back.', 500, raw)
  return { job_id: jobId, status: body.status }
}

export async function downloadPrediction(
  projectId: string,
  predictionId: string,
): Promise<{ blob: Blob; filename: string }> {
  const headers = new Headers()
  const res = await fetch(`/api/projects/${projectId}/predictions/${predictionId}/download`, {
    headers,
  })
  if (!res.ok) {
    let body: unknown = null
    const raw = await res.text()
    try {
      body = raw ? JSON.parse(raw) : null
    } catch {
      body = raw
    }
    throw new ApiError(extractErrorMessage(body, res.status), res.status, body)
  }
  const blob = await res.blob()
  const disposition = res.headers.get('content-disposition') ?? ''
  const match = /filename\*?=(?:UTF-8'')?["']?([^";]+)/i.exec(disposition)
  const filename = match ? decodeURIComponent(match[1]) : 'scored.csv'
  return { blob, filename }
}

export async function createConversation(projectId?: string): Promise<Conversation> {
  const raw = await request<unknown>('/api/conversations', {
    method: 'POST',
    body: JSON.stringify(projectId ? { project_id: projectId } : {}),
  })
  return (asObject<Conversation>(raw, ['conversation']) ?? raw) as Conversation
}

export async function listMessages(conversationId: string): Promise<ChatMessage[]> {
  const raw = await request<unknown>(`/api/conversations/${conversationId}/messages`)
  return asArray<ChatMessage>(raw, ['messages', 'items'])
}

type SseHandler = {
  onDelta?: (text: string) => void
  onEvent?: (event: Record<string, unknown>) => void
}

function applyChatEvent(
  payload: unknown,
  onDelta?: (text: string) => void,
): 'continue' | 'done' | 'error' | 'interrupted' {
  if (payload === '[DONE]') return 'done'
  if (typeof payload === 'string') {
    const trimmed = payload.trim()
    if (!trimmed) return 'continue'
    if (trimmed === '[DONE]') return 'done'
    try {
      return applyChatEvent(JSON.parse(trimmed), onDelta)
    } catch {
      onDelta?.(payload)
      return 'continue'
    }
  }
  if (!isRecord(payload)) return 'continue'
  const type = String(payload.type ?? payload.event ?? '').toLowerCase()
  if (type === 'done' || type === 'complete' || type === 'finished' || type === 'end') return 'done'
  if (type === 'interrupted' || type === 'abort' || payload.status === 'interrupted') {
    return 'interrupted'
  }
  if (type === 'error' || type === 'failed') {
    const message = str(payload.message)
    if (message && payload.status !== 'interrupted') onDelta?.(message)
    return 'error'
  }
  const delta =
    str(payload.text) ??
    str(payload.delta) ??
    str(payload.content) ??
    str(payload.token) ??
    str(payload.chunk) ??
    nestedDelta(payload)
  if (delta) onDelta?.(delta)
  if (payload.status === 'complete' && str(payload.content) && type === '') return 'done'
  return 'continue'
}

function nestedDelta(payload: Record<string, unknown>): string | undefined {
  const choices = payload.choices
  if (Array.isArray(choices) && isRecord(choices[0])) {
    const choice = choices[0]
    const delta = isRecord(choice.delta) ? str(choice.delta.content) : undefined
    const message = isRecord(choice.message) ? str(choice.message.content) : undefined
    return delta ?? message
  }
  if (isRecord(payload.message) && typeof payload.message.content === 'string') {
    return payload.message.content
  }
  return undefined
}

export type SseOutcome = 'done' | 'error' | 'interrupted'
export type SseEventResult = 'continue' | SseOutcome

function sseOutcome(result: SseEventResult): SseOutcome | null {
  if (result === 'continue') return null
  if (result === 'error') return 'error'
  if (result === 'interrupted') return 'interrupted'
  return 'done'
}

export async function readSse(
  res: Response,
  onEvent: (data: unknown, event: string | null) => SseEventResult,
  signal?: AbortSignal,
): Promise<SseOutcome> {
  if (!res.body) return 'interrupted'
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  let outcome: SseOutcome = 'interrupted'

  const cancel = () => {
    void reader.cancel()
  }
  signal?.addEventListener('abort', cancel, { once: true })

  try {
    while (true) {
      if (signal?.aborted) {
        return 'interrupted'
      }
      const { done, value } = await reader.read()
      if (done) {
        buf += decoder.decode()
        break
      }
      buf += decoder.decode(value, { stream: true })
      buf = buf.replace(/\r\n/g, '\n')
      let sep: number
      while ((sep = buf.indexOf('\n\n')) !== -1) {
        const block = buf.slice(0, sep)
        buf = buf.slice(sep + 2)
        const parsed = parseSseBlock(block)
        if (!parsed) continue
        const next = sseOutcome(onEvent(parsed.data, parsed.event))
        if (next) {
          outcome = next
          await reader.cancel()
          return signal?.aborted ? 'interrupted' : outcome
        }
      }
    }
    if (signal?.aborted) return 'interrupted'
    if (buf.trim()) {
      const parsed = parseSseBlock(buf)
      if (parsed) {
        const next = sseOutcome(onEvent(parsed.data, parsed.event))
        if (next) outcome = next
      }
    }
  } catch (err) {
    if (signal?.aborted || (err instanceof DOMException && err.name === 'AbortError')) {
      return 'interrupted'
    }
    throw err
  } finally {
    signal?.removeEventListener('abort', cancel)
  }
  return signal?.aborted ? 'interrupted' : outcome
}

function parseSseBlock(block: string): { event: string | null; data: unknown } | null {
  let event: string | null = null
  const dataLines: string[] = []
  for (const line of block.split('\n')) {
    if (!line || line.startsWith(':')) continue
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
  }
  if (!dataLines.length && !event) return null
  const raw = dataLines.join('\n')
  if (!raw) return { event, data: null }
  try {
    return { event, data: JSON.parse(raw) }
  } catch {
    return { event, data: raw }
  }
}

export async function sendChatMessage(
  conversationId: string,
  content: string,
  clientId: string,
  opts: SseHandler & { signal?: AbortSignal } = {},
): Promise<{ status: 'complete' | 'interrupted' | 'failed'; text: string }> {
  const headers = new Headers({
    'Content-Type': 'application/json',
    'X-Requested-With': REQUESTED_WITH,
    Accept: 'text/event-stream, application/json',
  })
  let res: Response
  try {
    res = await fetch(`/api/conversations/${conversationId}/messages`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ content, client_id: clientId }),
      signal: opts.signal,
    })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      return { status: 'interrupted', text: '' }
    }
    throw new ApiError(plainError(err), 0, null)
  }

  if (!res.ok) {
    let body: unknown = null
    const raw = await res.text()
    try {
      body = raw ? JSON.parse(raw) : null
    } catch {
      body = raw
    }
    throw new ApiError(extractErrorMessage(body, res.status), res.status, body)
  }

  const contentType = res.headers.get('content-type') ?? ''
  let assembled = ''
  const onDelta = (text: string) => {
    assembled += text
    opts.onDelta?.(text)
  }

  if (!contentType.includes('application/json')) {
    const outcome = await readSse(
      res,
      (data, event) => {
        if (isRecord(data)) opts.onEvent?.(data)
        const typed = event ? { ...(isRecord(data) ? data : { data }), type: event } : data
        return applyChatEvent(typed, onDelta)
      },
      opts.signal,
    )
    if (outcome === 'interrupted' || opts.signal?.aborted) {
      return { status: 'interrupted', text: assembled }
    }
    if (outcome === 'error') return { status: 'failed', text: assembled }
    return { status: 'complete', text: assembled }
  }

  const raw = await res.json()
  if (isRecord(raw)) opts.onEvent?.(raw)
  const text =
    str(isRecord(raw) ? raw.content : null) ??
    str(isRecord(raw) && isRecord(raw.message) ? raw.message.content : null) ??
    (typeof raw === 'string' ? raw : '')
  if (text) onDelta(text)
  const status = isRecord(raw) && raw.status === 'failed' ? 'failed' : 'complete'
  return { status, text: assembled || text }
}

export async function subscribeJobEvents(
  projectId: string,
  jobId: string,
  onEvent: (event: Record<string, unknown>) => void,
  signal?: AbortSignal,
): Promise<void> {
  const headers = new Headers({ Accept: 'text/event-stream' })
  let res: Response
  try {
    res = await fetch(`/api/projects/${projectId}/jobs/${jobId}/events`, {
      headers,
      signal,
    })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') return
    return
  }
  if (!res.ok) return
  await readSse(
    res,
    (data, event) => {
      if (isRecord(data)) {
        onEvent(event ? { ...data, type: String(data.type ?? event) } : data)
      }
      return 'continue'
    },
    signal,
  )
}

export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}
