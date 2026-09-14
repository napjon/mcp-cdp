import type {
  AppStatus,
  Budget,
  CardKey,
  ChatMessage,
  ColumnProfile,
  ColumnRole,
  DatasetDetail,
  DuplicateTimestamp,
  ExperimentConfig,
  Frequency,
  Job,
  MessageStatus,
  PreviewTable,
  Project,
  ReportPayload,
  SplitType,
  TaskType,
  ThreadItem,
} from './types'

export type AppState = {
  boot: 'loading' | 'ready'
  apiReachable: boolean
  bootNotice: string | null
  project: Project | null
  appStatus: AppStatus | null
  conversationId: string | null
  thread: ThreadItem[]
  composer: string
  sending: boolean
  dataset: DatasetDetail | null
  preview: PreviewTable | null
  columns: ColumnProfile[]
  dataError: string | null
  dataBusy: boolean
  task: TaskType | null
  target: string | null
  targetConfirmed: boolean
  dateColumn: string | null
  groupColumns: string[]
  frequency: Frequency
  horizon: number
  duplicateTimestamp: DuplicateTimestamp
  roles: Record<string, ColumnRole>
  columnsContinued: boolean
  split: SplitType
  testSize: number
  budget: Budget
  experimentId: string | null
  trainJob: Job | null
  trainError: string | null
  trainBusy: boolean
  report: ReportPayload | null
  modelId: string | null
  predictJob: Job | null
  predictError: string | null
  predictBusy: boolean
  predictionId: string | null
  expandedCard: CardKey | null
  allowExcludeInsufficient: boolean
}

export type Action =
  | { type: 'BOOT'; project: Project; appStatus: AppStatus | null; apiReachable: boolean; notice?: string | null }
  | { type: 'SET_CONVERSATION'; conversationId: string | null }
  | { type: 'LOAD_MESSAGES'; messages: ChatMessage[] }
  | { type: 'SET_COMPOSER'; value: string }
  | { type: 'SEND_START'; id: string; content: string; clientId: string; assistantId: string }
  | { type: 'ASSISTANT_DELTA'; id: string; text: string }
  | { type: 'ASSISTANT_END'; id: string; status: MessageStatus; text?: string }
  | { type: 'DATA_BUSY'; busy: boolean }
  | { type: 'DATA_ERROR'; error: string | null }
  | { type: 'DATA_LOADED'; dataset: DatasetDetail; preview: PreviewTable | null; columns: ColumnProfile[] }
  | { type: 'SET_TASK'; task: TaskType }
  | { type: 'SET_TARGET'; target: string }
  | { type: 'CONFIRM_TARGET' }
  | {
      type: 'SET_FORECAST'
      dateColumn?: string | null
      groupColumns?: string[]
      frequency?: Frequency
      horizon?: number
      duplicateTimestamp?: DuplicateTimestamp
    }
  | { type: 'SET_ROLE'; name: string; role: ColumnRole }
  | { type: 'CONTINUE_COLUMNS' }
  | { type: 'SET_SPLIT'; split: SplitType }
  | { type: 'SET_BUDGET'; budget: Budget }
  | { type: 'SET_ALLOW_EXCLUDE'; value: boolean }
  | { type: 'TRAIN_BUSY'; busy: boolean }
  | { type: 'TRAIN_ERROR'; error: string | null; jobId?: string }
  | { type: 'TRAIN_QUEUED'; experimentId: string; job: Job }
  | { type: 'JOB_UPDATE'; kind: 'train' | 'predict'; job: Job }
  | { type: 'REPORT_LOADED'; report: ReportPayload; modelId?: string | null; jobId?: string }
  | { type: 'PREDICT_BUSY'; busy: boolean }
  | { type: 'PREDICT_ERROR'; error: string | null; jobId?: string }
  | { type: 'PREDICT_QUEUED'; job: Job }
  | { type: 'PREDICT_READY'; predictionId: string; jobId?: string }
  | {
      type: 'RESTORE_RUN'
      experimentId?: string | null
      trainJob?: Job | null
      report?: ReportPayload | null
      modelId?: string | null
      predictJob?: Job | null
      predictionId?: string | null
    }
  | { type: 'TOGGLE_CARD'; card: CardKey }

const CARD_ORDER: CardKey[] = ['data', 'target', 'columns', 'confirm', 'report', 'predict']

export function initialState(): AppState {
  return {
    boot: 'loading',
    apiReachable: false,
    bootNotice: null,
    project: null,
    appStatus: null,
    conversationId: null,
    thread: [{ kind: 'card', id: 'card-data', card: 'data' }],
    composer: '',
    sending: false,
    dataset: null,
    preview: null,
    columns: [],
    dataError: null,
    dataBusy: false,
    task: null,
    target: null,
    targetConfirmed: false,
    dateColumn: null,
    groupColumns: [],
    frequency: 'D',
    horizon: 14,
    duplicateTimestamp: 'reject',
    roles: {},
    columnsContinued: false,
    split: 'random',
    testSize: 0.2,
    budget: 'quick',
    experimentId: null,
    trainJob: null,
    trainError: null,
    trainBusy: false,
    report: null,
    modelId: null,
    predictJob: null,
    predictError: null,
    predictBusy: false,
    predictionId: null,
    expandedCard: 'data',
    allowExcludeInsufficient: false,
  }
}

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'BOOT':
      return {
        ...state,
        boot: 'ready',
        project: action.project,
        appStatus: action.appStatus,
        apiReachable: action.apiReachable,
        bootNotice: action.notice ?? null,
      }
    case 'SET_CONVERSATION':
      return { ...state, conversationId: action.conversationId }
    case 'LOAD_MESSAGES': {
      const cards = state.thread.filter((item) => item.kind === 'card')
      const messages: ThreadItem[] = action.messages.map((msg) => ({
        kind: 'message' as const,
        id: msg.id,
        role: msg.role === 'user' ? 'user' : 'assistant',
        content: msg.content ?? '',
        status: (msg.status as MessageStatus) || 'complete',
        clientId: msg.client_id,
      }))
      return { ...state, thread: [...messages, ...cards] }
    }
    case 'SET_COMPOSER':
      return { ...state, composer: action.value }
    case 'SEND_START':
      return {
        ...state,
        composer: '',
        sending: true,
        thread: [
          ...state.thread,
          {
            kind: 'message',
            id: action.id,
            role: 'user',
            content: action.content,
            status: 'complete',
            clientId: action.clientId,
          },
          {
            kind: 'message',
            id: action.assistantId,
            role: 'assistant',
            content: '',
            status: 'streaming',
          },
        ],
      }
    case 'ASSISTANT_DELTA':
      return {
        ...state,
        thread: state.thread.map((item) =>
          item.kind === 'message' && item.id === action.id
            ? { ...item, content: item.content + action.text, status: 'streaming' }
            : item,
        ),
      }
    case 'ASSISTANT_END':
      return {
        ...state,
        sending: false,
        thread: state.thread.map((item) =>
          item.kind === 'message' && item.id === action.id
            ? {
                ...item,
                status: action.status,
                content: action.text != null && action.text.length ? action.text : item.content,
              }
            : item,
        ),
      }
    case 'DATA_BUSY':
      return {
        ...state,
        dataBusy: action.busy,
        dataError: action.busy ? null : state.dataError,
      }
    case 'DATA_ERROR':
      return { ...state, dataError: action.error, dataBusy: false }
    case 'DATA_LOADED': {
      const roles = defaultRoles(action.columns)
      const replacing = Boolean(state.dataset && state.dataset.id !== action.dataset.id)
      const base = replacing ? resetFrom(state, 'target') : state
      const next: AppState = {
        ...base,
        dataset: action.dataset,
        preview: action.preview,
        columns: action.columns,
        roles: replacing || !state.columns.length ? roles : state.roles,
        dataError: null,
        dataBusy: false,
        dateColumn: replacing || !state.dateColumn ? guessDateColumn(action.columns, roles) : state.dateColumn,
      }
      return ensureCard({ ...next, expandedCard: replacing || !state.targetConfirmed ? 'target' : next.expandedCard }, 'target')
    }
    case 'SET_TASK': {
      if (state.task === action.task) return state
      const split: SplitType = action.task === 'forecast' ? 'time' : state.split === 'time' ? 'random' : state.split
      const next = { ...state, task: action.task, split, targetConfirmed: false }
      return state.targetConfirmed || state.columnsContinued
        ? resetFrom(next, 'columns')
        : invalidateExperiment(next)
    }
    case 'SET_TARGET': {
      if (state.target === action.target) return state
      const next = { ...state, target: action.target, targetConfirmed: false }
      return state.targetConfirmed || state.columnsContinued
        ? resetFrom(next, 'columns')
        : invalidateExperiment(next)
    }
    case 'CONFIRM_TARGET': {
      if (!state.target || !state.task) return state
      if (state.task === 'forecast' && !state.dateColumn) return state
      const next = ensureCard({ ...state, targetConfirmed: true, expandedCard: 'columns' }, 'columns')
      return next
    }
    case 'SET_FORECAST': {
      const dateColumn = action.dateColumn !== undefined ? action.dateColumn : state.dateColumn
      const groupColumns = action.groupColumns ?? state.groupColumns
      const frequency = action.frequency ?? state.frequency
      const horizon = action.horizon ?? state.horizon
      const duplicateTimestamp = action.duplicateTimestamp ?? state.duplicateTimestamp
      if (
        dateColumn === state.dateColumn &&
        sameStringList(groupColumns, state.groupColumns) &&
        frequency === state.frequency &&
        horizon === state.horizon &&
        duplicateTimestamp === state.duplicateTimestamp
      ) {
        return state
      }
      return invalidateExperiment({
        ...state,
        dateColumn,
        groupColumns,
        frequency,
        horizon,
        duplicateTimestamp,
      })
    }
    case 'SET_ROLE': {
      if (state.roles[action.name] === action.role) return state
      const next = {
        ...state,
        roles: { ...state.roles, [action.name]: action.role },
        columnsContinued: false,
      }
      return state.columnsContinued ? resetFrom(next, 'confirm') : invalidateExperiment(next)
    }
    case 'CONTINUE_COLUMNS':
      return ensureCard({ ...state, columnsContinued: true, expandedCard: 'confirm' }, 'confirm')
    case 'SET_SPLIT':
      if (state.split === action.split) return state
      return invalidateExperiment({ ...state, split: action.split })
    case 'SET_BUDGET':
      if (state.budget === action.budget) return state
      return invalidateExperiment({ ...state, budget: action.budget })
    case 'SET_ALLOW_EXCLUDE':
      if (state.allowExcludeInsufficient === action.value) return state
      return invalidateExperiment({ ...state, allowExcludeInsufficient: action.value })
    case 'TRAIN_BUSY':
      return {
        ...state,
        trainBusy: action.busy,
        trainError: action.busy ? null : state.trainError,
      }
    case 'TRAIN_ERROR':
      if (action.jobId && state.trainJob?.id !== action.jobId) return state
      return { ...state, trainError: action.error, trainBusy: false }
    case 'TRAIN_QUEUED': {
      const next = {
        ...state,
        experimentId: action.experimentId,
        trainJob: action.job,
        trainError: null,
        trainBusy: false,
        report: null,
        modelId: null,
        predictionId: null,
        predictJob: null,
        predictError: null,
        expandedCard: 'report' as const,
        thread: dropCards(state.thread, 'predict'),
      }
      return ensureCard(next, 'report')
    }
    case 'JOB_UPDATE':
      if (action.kind === 'train') {
        if (!state.trainJob || state.trainJob.id !== action.job.id) return state
        return { ...state, trainJob: action.job }
      }
      if (!state.predictJob || state.predictJob.id !== action.job.id) return state
      return { ...state, predictJob: action.job }
    case 'REPORT_LOADED':
      if (!state.trainJob) return state
      if (action.jobId && action.jobId !== state.trainJob.id) return state
      return ensureCard(
        {
          ...state,
          report: action.report,
          modelId: action.modelId ?? state.modelId ?? action.report.model_id ?? null,
          expandedCard: 'report',
        },
        'predict',
      )
    case 'PREDICT_BUSY':
      return {
        ...state,
        predictBusy: action.busy,
        predictError: action.busy ? null : state.predictError,
      }
    case 'PREDICT_ERROR':
      if (action.jobId && state.predictJob?.id !== action.jobId) return state
      return { ...state, predictError: action.error, predictBusy: false }
    case 'PREDICT_QUEUED':
      return {
        ...state,
        predictJob: action.job,
        predictError: null,
        predictBusy: false,
        predictionId: null,
      }
    case 'PREDICT_READY':
      if (action.jobId && state.predictJob?.id !== action.jobId) return state
      return { ...state, predictionId: action.predictionId, predictBusy: false }
    case 'RESTORE_RUN': {
      let next: AppState = {
        ...state,
        experimentId: action.experimentId !== undefined ? action.experimentId : state.experimentId,
        trainJob: action.trainJob !== undefined ? action.trainJob : state.trainJob,
        report: action.report !== undefined ? action.report : state.report,
        modelId: action.modelId !== undefined ? action.modelId : state.modelId,
        predictJob: action.predictJob !== undefined ? action.predictJob : state.predictJob,
        predictionId: action.predictionId !== undefined ? action.predictionId : state.predictionId,
      }
      if (next.trainJob) next = ensureCard(next, 'report')
      if (next.report || next.modelId || next.predictJob || next.predictionId) {
        next = ensureCard(next, 'predict')
      }
      const trainStatus = next.trainJob ? String(next.trainJob.status) : ''
      const predictStatus = next.predictJob ? String(next.predictJob.status) : ''
      if (next.trainJob && (jobActive(trainStatus) || next.report)) {
        next = { ...next, expandedCard: 'report' }
      } else if (next.predictJob && (jobActive(predictStatus) || next.predictionId)) {
        next = { ...next, expandedCard: 'predict' }
      }
      return next
    }
    case 'TOGGLE_CARD':
      return {
        ...state,
        expandedCard: state.expandedCard === action.card ? null : action.card,
      }
    default:
      return state
  }
}

function ensureCard(state: AppState, card: CardKey): AppState {
  if (state.thread.some((item) => item.kind === 'card' && item.card === card)) return state
  return {
    ...state,
    thread: [...state.thread, { kind: 'card', id: `card-${card}`, card }],
  }
}

function dropCards(thread: ThreadItem[], from: CardKey): ThreadItem[] {
  const idx = CARD_ORDER.indexOf(from)
  const remove = new Set(CARD_ORDER.slice(idx))
  return thread.filter((item) => item.kind !== 'card' || !remove.has(item.card))
}

function resetFrom(state: AppState, from: CardKey): AppState {
  const thread = dropCards(state.thread, from)
  const next: AppState = { ...state, thread }
  const start = CARD_ORDER.indexOf(from)
  if (start <= CARD_ORDER.indexOf('target')) {
    next.task = null
    next.target = null
    next.targetConfirmed = false
    next.groupColumns = []
  }
  if (start <= CARD_ORDER.indexOf('columns')) {
    next.columnsContinued = false
  }
  if (start <= CARD_ORDER.indexOf('confirm')) {
    next.experimentId = null
    next.allowExcludeInsufficient = false
  }
  if (start <= CARD_ORDER.indexOf('report')) {
    next.trainJob = null
    next.trainError = null
    next.trainBusy = false
    next.report = null
    next.modelId = null
  }
  if (start <= CARD_ORDER.indexOf('predict')) {
    next.predictJob = null
    next.predictError = null
    next.predictBusy = false
    next.predictionId = null
  }
  return next
}

/** Drop stale train/predict artifacts without removing Confirm or the exclude checkbox. */
function invalidateExperiment(state: AppState): AppState {
  const next = resetFrom(state, 'report')
  next.experimentId = null
  return next
}

function sameStringList(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((item, index) => item === b[index])
}

export function isConstantColumn(col: ColumnProfile): boolean {
  return col.is_constant === 1 || col.is_constant === true
}

export function defaultRoleFor(col: ColumnProfile): ColumnRole {
  if (isConstantColumn(col)) return 'excluded'
  const inferred = col.user_role ?? col.inferred_role
  if (inferred === 'identifier') return 'excluded'
  return inferred
}

export function defaultRoles(columns: ColumnProfile[]): Record<string, ColumnRole> {
  const roles: Record<string, ColumnRole> = {}
  for (const col of columns) roles[col.name] = defaultRoleFor(col)
  return roles
}

export function excludeReason(col: ColumnProfile): string | null {
  if (isConstantColumn(col)) {
    return 'Every row has the same value, so this cannot help a prediction.'
  }
  if ((col.user_role ?? col.inferred_role) === 'identifier') {
    return 'This looks like an ID, not a real signal. Left out by default.'
  }
  return null
}

function guessDateColumn(columns: ColumnProfile[], roles: Record<string, ColumnRole>): string | null {
  const dated = columns.find((col) => roles[col.name] === 'date' || col.inferred_role === 'date')
  return dated?.name ?? null
}

export function buildConfig(state: AppState): ExperimentConfig | null {
  if (!state.task || !state.target) return null
  const excluded: string[] = []
  const features: string[] = []
  for (const col of state.columns) {
    if (col.name === state.target) continue
    if ((state.task === 'forecast' || state.split === 'time') && col.name === state.dateColumn) {
      continue
    }
    const role = state.roles[col.name] ?? defaultRoleFor(col)
    if (role === 'excluded' || role === 'identifier') excluded.push(col.name)
    else features.push(col.name)
  }
  return {
    task: state.task,
    target: state.target,
    features,
    excluded,
    roles: { ...state.roles },
    group_columns:
      state.task === 'forecast' || state.split === 'group' ? state.groupColumns : [],
    date_column:
      state.task === 'forecast' || state.split === 'time' ? state.dateColumn : null,
    frequency: state.task === 'forecast' ? state.frequency : null,
    horizon: state.task === 'forecast' ? state.horizon : null,
    duplicate_timestamp: state.duplicateTimestamp,
    entity_column: null,
    split: state.split,
    test_size: state.testSize,
    budget: state.budget,
    seed: 42,
    ...(state.allowExcludeInsufficient ? { allow_exclude_insufficient: true } : {}),
  }
}

export const MAX_FORECAST_SERIES = 100
export const HORIZON_MIN = 1
export const HORIZON_MAX = 90

export type SeriesEstimate = {
  count: number | null
  sampled: boolean
  overLimit: boolean
}

export function uniquePreviewCombos(
  preview: PreviewTable | null,
  groupColumns: string[],
): number | null {
  if (!groupColumns.length) return 1
  if (!preview?.columns.length || !preview.rows.length) return null
  const idxs = groupColumns.map((name) => preview.columns.indexOf(name))
  if (idxs.some((i) => i < 0)) return null
  const keys = new Set<string>()
  for (const row of preview.rows) {
    keys.add(idxs.map((i) => String(row[i] ?? '')).join('\u0000'))
  }
  return keys.size
}

export function estimateForecastSeries(
  groupColumns: string[],
  columns: ColumnProfile[],
  preview: PreviewTable | null,
): SeriesEstimate {
  if (!groupColumns.length) return { count: 1, sampled: false, overLimit: false }
  const uniques = groupColumns.map((name) => columns.find((col) => col.name === name)?.n_unique)
  const known = uniques.filter((n): n is number => typeof n === 'number')
  if (known.some((n) => n > MAX_FORECAST_SERIES)) {
    const count = groupColumns.length === 1 ? known[0] : Math.max(...known)
    return { count, sampled: false, overLimit: true }
  }
  if (groupColumns.length === 1 && known.length === 1) {
    return { count: known[0], sampled: false, overLimit: known[0] > MAX_FORECAST_SERIES }
  }
  const previewCount = uniquePreviewCombos(preview, groupColumns)
  if (previewCount == null) return { count: null, sampled: false, overLimit: false }
  const total = preview?.total
  const sampled = total != null && preview != null && preview.shown < total
  return {
    count: previewCount,
    sampled,
    overLimit: previewCount > MAX_FORECAST_SERIES,
  }
}

export function forecastSeriesNote(
  estimate: SeriesEstimate,
  groupColumns: string[],
): string | null {
  if (estimate.overLimit) {
    const n = estimate.count != null ? ` (${estimate.count.toLocaleString()})` : ''
    return `There are too many separate series to forecast${n}. The limit is ${MAX_FORECAST_SERIES}.`
  }
  if (!groupColumns.length) return null
  if (estimate.count == null) {
    return `Could not count series from this preview. Training stops above ${MAX_FORECAST_SERIES}.`
  }
  if (estimate.sampled) {
    return `Preview shows ${estimate.count.toLocaleString()} series; the limit is ${MAX_FORECAST_SERIES}.`
  }
  return null
}

export function insufficientHistoryGroups(texts: Array<string | null | undefined>): string[] {
  const groups: string[] = []
  const seen = new Set<string>()
  for (const raw of texts) {
    if (!raw) continue
    const idx = raw.search(/insufficient history for groups:/i)
    if (idx < 0) continue
    let rest = raw.slice(idx).replace(/^.*?groups:\s*/i, '')
    rest = rest.replace(/\.?\s*Set allow_exclude[\s\S]*$/i, '')
    const re = /(.+?)\s*\(need >=?\s*\d+\s*observations?\)/gi
    let match: RegExpExecArray | null
    let found = false
    while ((match = re.exec(rest))) {
      found = true
      const name = match[1].replace(/^,/, '').trim()
      if (name && !seen.has(name)) {
        seen.add(name)
        groups.push(name)
      }
    }
    if (!found) {
      const cleaned = rest.replace(/[.\s]+$/, '').trim()
      if (cleaned && !seen.has(cleaned)) {
        seen.add(cleaned)
        groups.push(cleaned)
      }
    }
  }
  return groups
}

export function mentionsInsufficientHistory(texts: Array<string | null | undefined>): boolean {
  return texts.some((text) => !!text && /insufficient history/i.test(text))
}

export function confirmWarnings(state: AppState): string[] {
  const warnings: string[] = []
  const config = buildConfig(state)
  const n = state.dataset?.current_version?.n_rows ?? state.dataset?.profile?.n_rows ?? state.preview?.total
  if (n != null && n < 30) {
    warnings.push('There are fewer than 30 rows, so the check on held-out rows will be shaky.')
  }
  if (config && config.features.length === 0) {
    warnings.push('No input columns are included. Add at least one column besides the target.')
  }
  const targetCol = state.columns.find((col) => col.name === state.target)
  if (targetCol && typeof targetCol.n_missing === 'number' && targetCol.n_missing > 0) {
    warnings.push(
      `${targetCol.n_missing.toLocaleString()} row${targetCol.n_missing === 1 ? '' : 's'} have no value in ${targetCol.name}. Those rows will be left out of training.`,
    )
  }
  if (targetCol && isConstantColumn(targetCol)) {
    warnings.push('The target is the same in every row, so there is nothing to learn.')
  }
  if (targetCol && (state.roles[targetCol.name] === 'identifier' || targetCol.inferred_role === 'identifier')) {
    warnings.push('The target looks like an ID. Predictions on IDs are rarely meaningful.')
  }
  if ((state.task === 'forecast' || state.split === 'time') && !state.dateColumn) {
    warnings.push('Pick the date column so we know the order of time.')
  }
  if (state.task === 'forecast' && (state.horizon < HORIZON_MIN || state.horizon > HORIZON_MAX)) {
    warnings.push(`Look-ahead must be between ${HORIZON_MIN} and ${HORIZON_MAX}.`)
  }
  if (state.task === 'forecast') {
    const series = estimateForecastSeries(state.groupColumns, state.columns, state.preview)
    const seriesNote = forecastSeriesNote(series, state.groupColumns)
    if (series.overLimit && seriesNote) warnings.push(seriesNote)
  }
  if (state.split === 'group' && state.groupColumns.length === 0) {
    warnings.push('“Keep groups together” needs a store, product, or similar column.')
  }
  if (state.dataset?.profile?.warnings) warnings.push(...state.dataset.profile.warnings)
  if (state.dataset?.warnings) warnings.push(...state.dataset.warnings)
  return warnings
}

export function jobActive(status: string | undefined): boolean {
  return status === 'queued' || status === 'running'
}


