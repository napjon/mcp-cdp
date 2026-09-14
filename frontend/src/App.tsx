import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useReducer,
  useRef,
  useState,
  type Dispatch,
  type MutableRefObject,
} from 'react'
import {
  attachPreviewTotal,
  cancelJob,
  columnsFromDataset,
  connectSheet,
  createConversation,
  createExperiment,
  createProject,
  downloadPrediction,
  getDataset,
  getJob,
  getPreview,
  getReport,
  getStatus,
  jobModelId,
  jobPredictionId,
  listMessages,
  listProjects,
  plainError,
  putColumns,
  refreshSheet,
  saveBlob,
  sendChatMessage,
  sheetTabGid,
  submitExperiment,
  submitPredict,
  subscribeJobEvents,
  uploadCsv,
} from './api'
import { Composer } from './components/Composer'
import { Header } from './components/Header'
import { MessageBubble } from './components/MessageBubble'
import { CardFrame } from './components/CardFrame'
import { ColumnsCard } from './components/cards/ColumnsCard'
import { ConfirmCard } from './components/cards/ConfirmCard'
import { DataCard } from './components/cards/DataCard'
import { PredictCard } from './components/cards/PredictCard'
import { ReportCard } from './components/cards/ReportCard'
import { TargetCard } from './components/cards/TargetCard'
import type { Action } from './workflow'
import type { CardKey, DatasetDetail, Job } from './types'
import { TASK_GOAL } from './types'
import {
  buildConfig,
  confirmWarnings,
  defaultRoleFor,
  estimateForecastSeries,
  forecastSeriesNote,
  initialState,
  jobActive,
  reducer,
} from './workflow'

const STORE_KEY = 'mcp-cdp.ui'
const PROJECT_NAME = 'Local workspace'

type Persist = {
  conversationId: string | null
  datasetId: string | null
  projectId: string | null
  experimentId: string | null
  trainJobId: string | null
  predictJobId: string | null
  modelId: string | null
  predictionId: string | null
}

function patchPersist(partial: Partial<Persist>) {
  try {
    const raw = sessionStorage.getItem(STORE_KEY)
    const cur = raw ? (JSON.parse(raw) as Persist) : {}
    sessionStorage.setItem(STORE_KEY, JSON.stringify({ ...cur, ...partial }))
  } catch {
    /* ignore quota */
  }
}

function loadPersist(): Persist | null {
  try {
    const raw = sessionStorage.getItem(STORE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<Persist>
    return {
      conversationId: parsed.conversationId ?? null,
      datasetId: parsed.datasetId ?? null,
      projectId: parsed.projectId ?? null,
      experimentId: parsed.experimentId ?? null,
      trainJobId: parsed.trainJobId ?? null,
      predictJobId: parsed.predictJobId ?? null,
      modelId: parsed.modelId ?? null,
      predictionId: parsed.predictionId ?? null,
    }
  } catch {
    return null
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function cardSummary(card: CardKey, state: ReturnType<typeof initialState>): string | undefined {
  if (card === 'data' && state.dataset) {
    const n = state.dataset.current_version?.n_rows ?? state.preview?.total
    return `${state.dataset.name}${n != null ? ` · ${n.toLocaleString()} rows` : ''}`
  }
  if (card === 'target' && state.targetConfirmed && state.task && state.target) {
    return `${state.target} · ${TASK_GOAL[state.task].title.toLowerCase()}`
  }
  if (card === 'columns' && state.columnsContinued) {
    const used = Object.entries(state.roles).filter(([, role]) => role !== 'excluded' && role !== 'identifier').length
    return `${used} columns included`
  }
  if (card === 'confirm' && state.trainJob) {
    return state.budget === 'quick' ? 'Quick training' : 'Thorough training'
  }
  if (card === 'report' && state.report) return 'Results are ready'
  if (card === 'predict' && state.predictionId) return 'Scored file ready'
  return undefined
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, undefined, initialState)
  const abortRef = useRef<AbortController | null>(null)
  const hydratedRef = useRef(false)
  const dataUploadingRef = useRef(false)
  const dataRequestRef = useRef(0)
  const [columnsError, setColumnsError] = useState<string | null>(null)
  const trainSuccessRef = useRef<(job: Job) => void>(() => {})
  const predictSuccessRef = useRef<(job: Job) => void>(() => {})

  useLayoutEffect(() => {
    if (!hydratedRef.current) return
    const persist: Persist = {
      conversationId: state.conversationId,
      datasetId: state.dataset?.id ?? null,
      projectId: state.project?.id ?? null,
      experimentId: state.experimentId,
      trainJobId: state.trainJob?.id ?? null,
      predictJobId: state.predictJob?.id ?? null,
      modelId: state.modelId,
      predictionId: state.predictionId,
    }
    try {
      sessionStorage.setItem(STORE_KEY, JSON.stringify(persist))
    } catch {
      /* ignore quota */
    }
  }, [
    state.conversationId,
    state.dataset?.id,
    state.project?.id,
    state.experimentId,
    state.trainJob?.id,
    state.predictJob?.id,
    state.modelId,
    state.predictionId,
  ])

  const loadReport = useCallback(async (projectId: string, job: Job) => {
    try {
      const report = await getReport(projectId, job.id)
      dispatch({
        type: 'REPORT_LOADED',
        report,
        modelId: jobModelId(job) ?? report.model_id ?? null,
        jobId: job.id,
      })
    } catch (err) {
      dispatch({ type: 'TRAIN_ERROR', error: plainError(err) })
    }
  }, [])

  const projectId = state.project?.id
  useEffect(() => {
    trainSuccessRef.current = (job) => {
      if (!projectId) return
      void loadReport(projectId, job)
    }
    predictSuccessRef.current = (job) => {
      const id = jobPredictionId(job)
      if (id) dispatch({ type: 'PREDICT_READY', predictionId: id, jobId: job.id })
      else {
        dispatch({
          type: 'PREDICT_ERROR',
          error: 'Scoring finished, but no download was returned.',
          jobId: job.id,
        })
      }
    }
  }, [projectId, loadReport])

  useEffect(() => {
    let cancelled = false
    async function boot() {
      const saved = loadPersist()
      let apiReachable = false
      let notice: string | null = null
      let appStatus = null
      try {
        appStatus = await getStatus()
        apiReachable = true
      } catch {
        notice = 'Can’t reach the local app. Training needs it running on this computer.'
      }

      let project = null
      try {
        const projects = await listProjects()
        project =
          (saved?.projectId ? projects.find((item) => item.id === saved.projectId) : undefined) ??
          projects.find((item) => item.name === PROJECT_NAME) ??
          projects[0] ??
          null
        if (!project) project = await createProject(PROJECT_NAME)
        apiReachable = true
      } catch (err) {
        if (!notice) notice = plainError(err)
      }

      if (cancelled) return
      if (!project) {
        dispatch({
          type: 'BOOT',
          project: { id: '', name: PROJECT_NAME },
          appStatus,
          apiReachable,
          notice,
        })
        hydratedRef.current = true
        return
      }

      dispatch({ type: 'BOOT', project, appStatus, apiReachable, notice })

      let conversationId = saved?.conversationId ?? null
      if (conversationId) {
        try {
          const messages = await listMessages(conversationId)
          if (!cancelled) {
            dispatch({ type: 'SET_CONVERSATION', conversationId })
            dispatch({ type: 'LOAD_MESSAGES', messages })
            patchPersist({ conversationId })
          }
        } catch {
          conversationId = null
        }
      }
      if (!conversationId) {
        try {
          const conversation = await createConversation(project.id)
          conversationId = conversation.id
          if (!cancelled) {
            dispatch({ type: 'SET_CONVERSATION', conversationId })
            patchPersist({ conversationId })
          }
        } catch {
          /* Chat is optional; cards still work. */
        }
      }

      if (saved?.datasetId) {
        try {
          const dataset = await getDataset(project.id, saved.datasetId)
          const preview = await getPreview(project.id, saved.datasetId, 20).catch(() => null)
          if (!cancelled) {
            dispatch({
              type: 'DATA_LOADED',
              dataset,
              preview: attachPreviewTotal(preview, dataset),
              columns: columnsFromDataset(dataset, preview),
            })
          }
        } catch {
          /* start from empty data card */
        }
      }

      if (saved && project && !cancelled) {
        await restoreRun(project.id, saved, () => cancelled, dispatch)
      }

      if (!cancelled) {
        hydratedRef.current = true
        patchPersist({ conversationId, projectId: project.id })
      }
    }
    void boot()
    return () => {
      cancelled = true
    }
  }, [])

  useJobWatcher(state.project?.id, state.trainJob, 'train', dispatch, trainSuccessRef)
  useJobWatcher(state.project?.id, state.predictJob, 'predict', dispatch, predictSuccessRef)

  async function ingestDataset(loader: () => Promise<DatasetDetail>) {
    if (dataUploadingRef.current || state.dataBusy) return
    if (!state.project?.id) {
      dispatch({ type: 'DATA_ERROR', error: 'The local app is not reachable yet.' })
      return
    }
    dataUploadingRef.current = true
    const requestId = ++dataRequestRef.current
    dispatch({ type: 'DATA_ERROR', error: null })
    dispatch({ type: 'DATA_BUSY', busy: true })
    setColumnsError(null)
    try {
      const uploaded = await loader()
      if (requestId !== dataRequestRef.current) return
      const dataset = await getDataset(state.project.id, uploaded.id).catch(() => uploaded)
      if (requestId !== dataRequestRef.current) return
      const preview = await getPreview(state.project.id, dataset.id, 20).catch(() => null)
      if (requestId !== dataRequestRef.current) return
      dispatch({
        type: 'DATA_LOADED',
        dataset,
        preview: attachPreviewTotal(preview, dataset),
        columns: columnsFromDataset(dataset, preview),
      })
    } catch (err) {
      if (requestId !== dataRequestRef.current) return
      dispatch({ type: 'DATA_ERROR', error: plainError(err) })
    } finally {
      if (requestId === dataRequestRef.current) dataUploadingRef.current = false
    }
  }

  async function onSend() {
    const content = state.composer.trim()
    if (!content || state.sending) return
    const userId = crypto.randomUUID()
    const assistantId = crypto.randomUUID()
    const clientId = crypto.randomUUID()
    dispatch({ type: 'SEND_START', id: userId, content, clientId, assistantId })

    let conversationId = state.conversationId
    if (!conversationId && state.project?.id) {
      try {
        const conversation = await createConversation(state.project.id)
        conversationId = conversation.id
        dispatch({ type: 'SET_CONVERSATION', conversationId })
        patchPersist({ conversationId })
      } catch {
        conversationId = null
      }
    }

    if (!conversationId) {
      dispatch({
        type: 'ASSISTANT_END',
        id: assistantId,
        status: 'complete',
        text: 'Chat answers are limited right now. You can still add a spreadsheet and train from the cards.',
      })
      return
    }

    const ac = new AbortController()
    abortRef.current = ac
    try {
      const result = await sendChatMessage(conversationId, content, clientId, {
        signal: ac.signal,
        onDelta: (text) => dispatch({ type: 'ASSISTANT_DELTA', id: assistantId, text }),
      })
      const fallback =
        result.text
          ? undefined
          : result.status === 'interrupted'
            ? 'Stopped.'
            : result.status === 'complete'
              ? 'Chat answers are limited right now. You can still add a spreadsheet and train from the cards.'
              : 'The reply did not finish. You can still train from the cards.'
      dispatch({
        type: 'ASSISTANT_END',
        id: assistantId,
        status: result.status,
        text: result.text || fallback,
      })
    } catch (err) {
      dispatch({
        type: 'ASSISTANT_END',
        id: assistantId,
        status: 'failed',
        text: `${plainError(err)} You can still train from the cards.`,
      })
    } finally {
      abortRef.current = null
    }
  }

  function onStop() {
    abortRef.current?.abort()
  }

  async function onTrain() {
    const config = buildConfig(state)
    if (!state.project?.id || !state.dataset || !config) return
    dispatch({ type: 'TRAIN_BUSY', busy: true })
    try {
      await putColumns(
        state.project.id,
        state.dataset.id,
        state.columns.map((col) => ({
          name: col.name,
          role: state.roles[col.name] ?? defaultRoleFor(col),
        })),
      )
      const experiment = await createExperiment(state.project.id, {
        dataset_id: state.dataset.id,
        config,
      })
      const submitted = await submitExperiment(state.project.id, experiment.id)
      dispatch({
        type: 'TRAIN_QUEUED',
        experimentId: experiment.id,
        job: {
          id: submitted.job_id,
          type: 'train',
          status: submitted.status ?? 'queued',
        },
      })
      patchPersist({
        projectId: state.project.id,
        datasetId: state.dataset.id,
        experimentId: experiment.id,
        trainJobId: submitted.job_id,
      })
    } catch (err) {
      dispatch({ type: 'TRAIN_ERROR', error: plainError(err) })
    }
  }

  async function onScore(file: File) {
    if (!state.project?.id) return
    dispatch({ type: 'PREDICT_BUSY', busy: true })
    try {
      const submitted = await submitPredict(state.project.id, {
        file,
        model_id: state.modelId ?? undefined,
        dataset_id: state.dataset?.id,
      })
      dispatch({
        type: 'PREDICT_QUEUED',
        job: { id: submitted.job_id, type: 'predict', status: submitted.status ?? 'queued' },
      })
    } catch (err) {
      dispatch({ type: 'PREDICT_ERROR', error: plainError(err) })
    }
  }

  async function onDownload() {
    if (!state.project?.id || !state.predictionId) return
    try {
      const { blob, filename } = await downloadPrediction(state.project.id, state.predictionId)
      saveBlob(blob, filename)
    } catch (err) {
      dispatch({ type: 'PREDICT_ERROR', error: plainError(err) })
    }
  }

  const config = buildConfig(state)
  const showWelcome = state.thread.every((item) => item.kind !== 'message')
  const seriesEst =
    state.task === 'forecast'
      ? estimateForecastSeries(state.groupColumns, state.columns, state.preview)
      : null
  const seriesNote = seriesEst ? forecastSeriesNote(seriesEst, state.groupColumns) : null

  return (
    <div className="app">
      <a className="skip" href="#composer-input">
        Skip to message
      </a>
      <Header
        projectName={state.project?.name || PROJECT_NAME}
        appStatus={state.appStatus}
        apiReachable={state.apiReachable}
        bootNotice={state.bootNotice}
      />
      <div className="thread-wrap">
        <div className="thread">
          {showWelcome ? (
            <div className="welcome">
              <h2>Find patterns in a spreadsheet</h2>
              <p>
                Add a CSV or a public Google Sheet. Then we will walk through what to predict, which
                columns to use, and how well a model does — no chat account required.
              </p>
              {state.boot === 'loading' ? (
                <p className="caption" role="status">
                  Starting…
                </p>
              ) : null}
            </div>
          ) : null}

          {state.thread.map((item) => {
            if (item.kind === 'message') {
              return <MessageBubble key={item.id} item={item} />
            }
            const expanded = state.expandedCard === item.card
            const complete =
              (item.card === 'data' && Boolean(state.dataset)) ||
              (item.card === 'target' && state.targetConfirmed) ||
              (item.card === 'columns' && state.columnsContinued) ||
              (item.card === 'confirm' && Boolean(state.trainJob)) ||
              (item.card === 'report' && Boolean(state.report)) ||
              (item.card === 'predict' && Boolean(state.predictionId))
            const titles: Record<CardKey, string> = {
              data: 'Data',
              target: 'What to predict',
              columns: 'Columns',
              confirm: 'Confirm & train',
              report: 'How good is it?',
              predict: 'Predict',
            }
            const steps: Record<CardKey, number> = {
              data: 1,
              target: 2,
              columns: 3,
              confirm: 4,
              report: 5,
              predict: 6,
            }
            return (
              <CardFrame
                key={item.id}
                step={steps[item.card]}
                title={titles[item.card]}
                summary={cardSummary(item.card, state)}
                expanded={expanded}
                complete={complete}
                onToggle={() => dispatch({ type: 'TOGGLE_CARD', card: item.card })}
              >
                {item.card === 'data' ? (
                  <DataCard
                    key={`${state.project?.id ?? 'none'}:${state.dataset?.id ?? 'empty'}`}
                    projectId={state.project?.id ?? ''}
                    dataset={state.dataset}
                    preview={state.preview}
                    busy={state.dataBusy}
                    error={state.dataError}
                    onUpload={(file) => {
                      if (state.dataBusy) return
                      void ingestDataset(() => uploadCsv(state.project!.id, file))
                    }}
                    onSheet={(url, headerRow) => {
                      void ingestDataset(() =>
                        connectSheet(state.project!.id, {
                          url,
                          header_row: headerRow,
                          gid: sheetTabGid(url) ?? undefined,
                        }),
                      )
                    }}
                    onRefresh={
                      state.dataset?.connection_id
                        ? () => {
                            void ingestDataset(async () => {
                              await refreshSheet(state.project!.id, state.dataset!.connection_id as string)
                              return getDataset(state.project!.id, state.dataset!.id)
                            })
                          }
                        : undefined
                    }
                  />
                ) : null}
                {item.card === 'target' ? (
                  <TargetCard
                    columns={state.columns}
                    task={state.task}
                    target={state.target}
                    dateColumn={state.dateColumn}
                    groupColumns={state.groupColumns}
                    frequency={state.frequency}
                    horizon={state.horizon}
                    duplicateTimestamp={state.duplicateTimestamp}
                    confirmed={state.targetConfirmed}
                    seriesOverLimit={seriesEst?.overLimit ?? false}
                    seriesNote={seriesNote}
                    onTask={(task) => dispatch({ type: 'SET_TASK', task })}
                    onTarget={(target) => dispatch({ type: 'SET_TARGET', target })}
                    onForecast={(patch) => dispatch({ type: 'SET_FORECAST', ...patch })}
                    onConfirm={() => dispatch({ type: 'CONFIRM_TARGET' })}
                  />
                ) : null}
                {item.card === 'columns' ? (
                  <ColumnsCard
                    columns={state.columns}
                    roles={state.roles}
                    target={state.target}
                    dateColumn={state.dateColumn}
                    error={columnsError}
                    onRole={(name, role) => dispatch({ type: 'SET_ROLE', name, role })}
                    onContinue={() => {
                      void (async () => {
                        setColumnsError(null)
                        if (state.project?.id && state.dataset) {
                          try {
                            await putColumns(
                              state.project.id,
                              state.dataset.id,
                              state.columns.map((col) => ({
                                name: col.name,
                                role: state.roles[col.name] ?? defaultRoleFor(col),
                              })),
                            )
                          } catch (err) {
                            setColumnsError(plainError(err))
                            return
                          }
                        }
                        dispatch({ type: 'CONTINUE_COLUMNS' })
                      })()
                    }}
                  />
                ) : null}
                {item.card === 'confirm' ? (
                  <ConfirmCard
                    config={config}
                    task={state.task}
                    columns={state.columns}
                    warnings={confirmWarnings(state)}
                    seriesNote={seriesNote}
                    seriesOverLimit={seriesEst?.overLimit ?? false}
                    busy={state.trainBusy || jobActive(state.trainJob?.status as string | undefined)}
                    error={state.trainError ?? state.trainJob?.error ?? null}
                    allowExcludeInsufficient={state.allowExcludeInsufficient}
                    onAllowExclude={(value) => dispatch({ type: 'SET_ALLOW_EXCLUDE', value })}
                    onSplit={(split) => dispatch({ type: 'SET_SPLIT', split })}
                    onBudget={(budget) => dispatch({ type: 'SET_BUDGET', budget })}
                    onDateColumn={(dateColumn) => dispatch({ type: 'SET_FORECAST', dateColumn })}
                    onGroupColumns={(groupColumns) => dispatch({ type: 'SET_FORECAST', groupColumns })}
                    onTrain={() => {
                      void onTrain()
                    }}
                  />
                ) : null}
                {item.card === 'report' ? (
                  <ReportCard
                    projectId={state.project?.id ?? ''}
                    job={state.trainJob}
                    report={state.report}
                    error={state.trainError}
                    onCancel={() => {
                      if (state.project?.id && state.trainJob) {
                        void cancelJob(state.project.id, state.trainJob.id).catch((err) => {
                          dispatch({ type: 'TRAIN_ERROR', error: plainError(err) })
                        })
                      }
                    }}
                  />
                ) : null}
                {item.card === 'predict' ? (
                  <PredictCard
                    ready={Boolean(state.modelId || (state.trainJob && state.trainJob.status === 'succeeded'))}
                    busy={state.predictBusy}
                    error={state.predictError}
                    job={state.predictJob}
                    predictionId={state.predictionId}
                    onScore={(file) => {
                      void onScore(file)
                    }}
                    onDownload={() => {
                      void onDownload()
                    }}
                    onCancel={() => {
                      if (state.project?.id && state.predictJob) {
                        void cancelJob(state.project.id, state.predictJob.id).catch((err) => {
                          dispatch({ type: 'PREDICT_ERROR', error: plainError(err) })
                        })
                      }
                    }}
                  />
                ) : null}
              </CardFrame>
            )
          })}
        </div>
      </div>
      <Composer
        value={state.composer}
        sending={state.sending}
        onChange={(value) => dispatch({ type: 'SET_COMPOSER', value })}
        onSend={() => {
          void onSend()
        }}
        onStop={onStop}
      />
    </div>
  )
}

async function restoreRun(
  projectId: string,
  saved: Persist,
  cancelled: () => boolean,
  dispatch: Dispatch<Action>,
) {
  if (!saved.trainJobId && !saved.predictJobId && !saved.modelId && !saved.predictionId) return
  let trainJob: Job | null = null
  let predictJob: Job | null = null
  if (saved.trainJobId) {
    try {
      trainJob = await getJob(projectId, saved.trainJobId)
    } catch {
      trainJob = null
    }
  }
  if (cancelled()) return
  if (saved.predictJobId) {
    try {
      predictJob = await getJob(projectId, saved.predictJobId)
    } catch {
      predictJob = null
    }
  }
  if (cancelled()) return

  const trainStatus = trainJob ? String(trainJob.status) : ''
  const predictStatus = predictJob ? String(predictJob.status) : ''
  let report = null
  let reportError: string | null = null
  if (trainJob && trainStatus === 'succeeded') {
    try {
      report = await getReport(projectId, trainJob.id)
    } catch (err) {
      reportError = plainError(err)
    }
  }
  if (cancelled()) return

  const predictionId =
    predictJob && predictStatus === 'succeeded'
      ? jobPredictionId(predictJob) ?? saved.predictionId
      : predictJob && jobActive(predictStatus)
        ? null
        : saved.predictionId

  dispatch({
    type: 'RESTORE_RUN',
    experimentId: saved.experimentId,
    trainJob,
    report,
    modelId: (trainJob ? jobModelId(trainJob) : null) ?? saved.modelId,
    predictJob,
    predictionId,
  })

  if (trainJob && trainStatus === 'failed') {
    dispatch({
      type: 'TRAIN_ERROR',
      error: trainJob.error || 'Training failed.',
      jobId: trainJob.id,
    })
  } else if (reportError && trainJob) {
    dispatch({ type: 'TRAIN_ERROR', error: reportError, jobId: trainJob.id })
  }
  if (predictJob && predictStatus === 'failed') {
    dispatch({
      type: 'PREDICT_ERROR',
      error: predictJob.error || 'Scoring failed.',
      jobId: predictJob.id,
    })
  }
}

function useJobWatcher(
  projectId: string | undefined,
  job: Job | null,
  kind: 'train' | 'predict',
  dispatch: Dispatch<Action>,
  onSuccessRef: MutableRefObject<(job: Job) => void>,
) {
  const jobId = job?.id
  const status = job ? String(job.status) : ''

  useEffect(() => {
    if (!projectId || !jobId || !jobActive(status)) return
    const pid = projectId
    const jid = jobId
    const ac = new AbortController()
    let stop = false
    let finished = false

    function finish(latest: Job) {
      if (finished) return
      finished = true
      stop = true
      void (async () => {
        let canonical = latest
        try {
          canonical = await getJob(pid, jid)
        } catch {
          /* use the event snapshot */
        }
        dispatch({ type: 'JOB_UPDATE', kind, job: canonical })
        if (canonical.status === 'succeeded') onSuccessRef.current(canonical)
        if (canonical.status === 'failed') {
          dispatch({
            type: kind === 'train' ? 'TRAIN_ERROR' : 'PREDICT_ERROR',
            error: canonical.error || (kind === 'train' ? 'Training failed.' : 'Scoring failed.'),
            jobId: canonical.id,
          })
        }
      })()
    }

    void subscribeJobEvents(
      pid,
      jid,
      (event) => {
        const nextStatus = typeof event.status === 'string' ? event.status : status
        const progress =
          typeof event.progress === 'object' && event.progress !== null
            ? (event.progress as Job['progress'])
            : undefined
        const latest: Job = {
          id: String(event.job_id ?? event.id ?? jobId),
          type: kind,
          status: nextStatus,
          progress_json: progress ?? event,
          progress,
          error: typeof event.error === 'string' ? event.error : null,
          model_id: typeof event.model_id === 'string' ? event.model_id : null,
          prediction_id: typeof event.prediction_id === 'string' ? event.prediction_id : null,
        }
        dispatch({ type: 'JOB_UPDATE', kind, job: latest })
        if (!jobActive(String(latest.status))) finish(latest)
      },
      ac.signal,
    )

    async function poll() {
      while (!stop) {
        try {
          const latest = await getJob(pid, jid)
          dispatch({ type: 'JOB_UPDATE', kind, job: latest })
          if (!jobActive(String(latest.status))) {
            finish(latest)
            break
          }
        } catch {
          /* keep polling through brief errors */
        }
        await sleep(1500)
      }
    }
    void poll()
    return () => {
      stop = true
      ac.abort()
    }
  }, [projectId, jobId, status, kind, dispatch, onSuccessRef])
}
