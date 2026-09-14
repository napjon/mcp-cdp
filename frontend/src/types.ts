export type TaskType = 'classification' | 'regression' | 'forecast'
export type ColumnRole =
  | 'numeric'
  | 'categorical'
  | 'text'
  | 'date'
  | 'identifier'
  | 'excluded'
export type SplitType = 'random' | 'group' | 'time'
export type Budget = 'quick' | 'thorough'
export type JobType = 'train' | 'predict'
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled'
export type MessageStatus = 'complete' | 'streaming' | 'interrupted' | 'failed'
export type SourceType = 'csv' | 'sheets'
export type Frequency = 'D' | 'W' | 'M'
export type DuplicateTimestamp = 'reject' | 'aggregate_mean'

export type AppStatus = {
  llm_configured: boolean
  provider: string | null
  mcp_token_set: boolean
}

export type Project = {
  id: string
  name: string
  created_at?: string
}

export type DatasetVersion = {
  id: string
  dataset_id?: string
  version: number
  content_hash?: string
  original_path?: string
  normalized_path?: string
  n_rows?: number
  n_cols?: number
  header_row?: number
  encoding?: string
  delimiter?: string
  created_at?: string
}

export type ColumnProfile = {
  id?: string
  version_id?: string
  name: string
  inferred_role: ColumnRole
  user_role?: ColumnRole | null
  n_missing?: number
  n_unique?: number
  is_constant?: number | boolean
  sample_preview?: string | null
}

export type Dataset = {
  id: string
  project_id?: string
  name: string
  source_type: SourceType
  connection_id?: string | null
  created_at?: string
}

export type DatasetDetail = Dataset & {
  current_version?: DatasetVersion | null
  version?: DatasetVersion | null
  columns?: ColumnProfile[]
  profile?: {
    n_rows?: number
    n_cols?: number
    columns?: ColumnProfile[]
    warnings?: string[]
    summary?: string
  }
  warnings?: string[]
}

export type PreviewTable = {
  columns: string[]
  rows: string[][]
  shown: number
  total?: number
}

export type ExperimentConfig = {
  task: TaskType
  target: string
  features: string[]
  excluded: string[]
  roles: Record<string, ColumnRole>
  group_columns: string[]
  date_column: string | null
  frequency: Frequency | null
  horizon: number | null
  duplicate_timestamp: DuplicateTimestamp
  entity_column: string | null
  split: SplitType
  test_size: number
  budget: Budget
  seed: number
  allow_exclude_insufficient?: boolean
}

export type Experiment = {
  id: string
  project_id?: string
  dataset_id?: string
  created_at?: string
}

export type JobProgress = {
  percent?: number
  progress?: number
  step?: string
  message?: string
  stage?: string
  prediction_id?: string
  model_id?: string
}

export type Job = {
  id: string
  project_id?: string
  experiment_revision_id?: string | null
  model_id?: string | null
  prediction_id?: string | null
  type: JobType | string
  status: JobStatus | string
  progress_json?: string | JobProgress | null
  progress?: JobProgress | Record<string, unknown> | null
  error?: string | null
  created_at?: string
  started_at?: string | null
  finished_at?: string | null
  result?: {
    prediction_id?: string
    model_id?: string
    report_id?: string
  }
}

export type PlotRef = {
  name?: string
  title?: string
  url?: string
  path?: string
  filename?: string
  kind?: string
}

export type SeriesMetrics = {
  series?: string
  group?: string
  name?: string
  metrics?: Record<string, number | string>
}

export type ReportPayload = {
  task?: TaskType | string
  dataset_version?: number | string
  dataset_version_id?: string
  split?: string
  n?: number
  n_rows?: number
  n_test?: number
  n_val?: number
  n_validation?: number
  metrics?: Record<string, unknown>
  units?: string | null
  baseline_metrics?: Record<string, unknown>
  baseline?: Record<string, unknown>
  worse_than_baseline?: boolean
  selected_candidate?: string
  warnings?: string[]
  limitation?: string
  plots?: Array<PlotRef | string>
  plot_files?: Array<PlotRef | string>
  confusion_matrix?: PlotRef | string
  residuals?: PlotRef | string
  per_series?: SeriesMetrics[]
  aggregate?: Record<string, number | string>
  model_id?: string
}

export type Report = {
  id?: string
  job_id: string
  report_json?: ReportPayload | string
  plot_dir?: string
  metrics?: Record<string, number | string>
  plots?: Array<PlotRef | string>
}

export type Conversation = {
  id: string
  project_id?: string
  created_at?: string
}

export type ChatMessage = {
  id: string
  conversation_id?: string
  role: 'user' | 'assistant' | string
  content: string
  client_id?: string
  status?: MessageStatus | string
  created_at?: string
}

export type SubmitJobResponse = {
  job_id: string
  status?: string
}

export type SampleDisclosure = {
  dataset_id?: string
  enabled: boolean
  previewed_at?: string | null
  columns?: string[]
  rows?: string[][] | Record<string, unknown>[]
  sample_rows?: number
  ok?: boolean
  error?: string
}

export type CardKey =
  | 'data'
  | 'target'
  | 'columns'
  | 'confirm'
  | 'report'
  | 'predict'

export type ThreadMessage = {
  kind: 'message'
  id: string
  role: 'user' | 'assistant'
  content: string
  status: MessageStatus
  clientId?: string
}

export type ThreadCard = {
  kind: 'card'
  id: string
  card: CardKey
}

export type ThreadItem = ThreadMessage | ThreadCard

export const COLUMN_ROLES: ColumnRole[] = [
  'numeric',
  'categorical',
  'text',
  'date',
  'identifier',
  'excluded',
]

export const ROLE_LABEL: Record<ColumnRole, string> = {
  numeric: 'number',
  categorical: 'category',
  text: 'text',
  date: 'date',
  identifier: 'id',
  excluded: 'left out',
}

export const TASK_GOAL: Record<TaskType, { title: string; blurb: string; example: string }> =
  {
    classification: {
      title: 'Category',
      blurb: 'Sort each row into a group.',
      example: 'Will this customer cancel? Yes or no.',
    },
    regression: {
      title: 'Number',
      blurb: 'Estimate a value for each row.',
      example: 'How much might this house sell for?',
    },
    forecast: {
      title: 'Future values',
      blurb: 'Look ahead over time, by store or product if you have them.',
      example: 'How many units will each store sell next week?',
    },
  }
