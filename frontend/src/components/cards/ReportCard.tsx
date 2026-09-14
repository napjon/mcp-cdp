import { useState } from 'react'
import { flattenMetrics, parseProgress, plotUrl } from '../../api'
import type { Job, PlotRef, ReportPayload, SeriesMetrics, TaskType } from '../../types'
import { jobActive } from '../../workflow'

type Props = {
  projectId: string
  job: Job | null
  report: ReportPayload | null
  error: string | null
  onCancel: () => void
}

const METRIC_LABELS: Record<string, string> = {
  macro_f1: 'Macro-F1',
  'macro-f1': 'Macro-F1',
  f1_macro: 'Macro-F1',
  'f1-macro': 'Macro-F1',
  balanced_accuracy: 'Balanced accuracy',
  balanced_acc: 'Balanced accuracy',
  accuracy: 'Accuracy',
  mae: 'MAE',
  rmse: 'RMSE',
  val_mae: 'Validation MAE',
  val_rmse: 'Validation RMSE',
  val_score: 'Validation score',
  r2: 'R²',
  r_squared: 'R²',
  mape: 'MAPE',
  smape: 'sMAPE',
  score: 'Score',
}

const UNIT_METRICS = new Set(['mae', 'rmse', 'val_mae', 'val_rmse'])

const HIGHER_BETTER = new Set([
  'macro_f1',
  'macro-f1',
  'f1_macro',
  'f1-macro',
  'balanced_accuracy',
  'balanced_acc',
  'accuracy',
  'r2',
  'r_squared',
])

function humanize(key: string): string {
  return METRIC_LABELS[key] ?? key.replace(/_/g, ' ')
}

function formatNumber(value: number | string): string {
  if (typeof value === 'string') return value
  if (!Number.isFinite(value)) return '—'
  const abs = Math.abs(value)
  if (abs >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 1 })
  if (abs >= 10) return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
  return value.toLocaleString(undefined, { maximumFractionDigits: 3 })
}

function metricBag(report: ReportPayload): Record<string, number | string> {
  const nested = flattenMetrics(report.metrics)
  const extra = flattenMetrics(report.aggregate)
  return { ...extra, ...nested }
}

function scalarBag(raw: unknown): Record<string, number | string> {
  if (!isRecord(raw)) return {}
  const out: Record<string, number | string> = {}
  for (const [key, value] of Object.entries(raw)) {
    if (typeof value === 'number' || typeof value === 'string') out[key] = value
  }
  return out
}

function splitBags(report: ReportPayload): {
  test: Record<string, number | string> | null
  validation: Record<string, number | string> | null
} {
  const root = isRecord(report.metrics) ? report.metrics : null
  const testNested = root && isRecord(root.test) ? scalarBag(root.test) : null
  const valNested = root && isRecord(root.validation) ? scalarBag(root.validation) : null
  if (testNested || valNested) {
    return {
      test: testNested && Object.keys(testNested).length ? testNested : null,
      validation: valNested && Object.keys(valNested).length ? valNested : null,
    }
  }
  const flat = metricBag(report)
  return { test: Object.keys(flat).length ? flat : null, validation: null }
}

function pickMetrics(
  source: Record<string, number | string>,
  task: string,
  opts: { keepScore?: boolean } = {},
): Array<[string, number | string]> {
  const preferred =
    task === 'classification'
      ? ['macro_f1', 'macro-f1', 'f1_macro', 'balanced_accuracy', 'balanced_acc']
      : task === 'regression'
        ? ['mae', 'rmse', 'r2', 'r_squared']
        : ['mae', 'rmse', 'mape', 'smape', 'r2']
  const seen = new Set<string>()
  const rows: Array<[string, number | string]> = []
  for (const key of preferred) {
    if (source[key] != null && !seen.has(humanize(key))) {
      seen.add(humanize(key))
      rows.push([key, source[key]])
    }
  }
  for (const [key, value] of Object.entries(source)) {
    if (rows.length >= 6) break
    if (key === 'seed' || key === 'n' || key === 'task') continue
    if (key === 'score' && !opts.keepScore) continue
    if (!seen.has(humanize(key))) {
      seen.add(humanize(key))
      rows.push([key, value])
    }
  }
  return rows
}

function perClassRows(
  report: ReportPayload,
): Array<{ label: string; precision: number | string | undefined; recall: number | string | undefined }> {
  const root = isRecord(report.metrics) ? report.metrics : null
  const test = root && isRecord(root.test) ? root.test : root
  const raw = (isRecord(test) && test.per_class) || (root && root.per_class)
  if (!isRecord(raw)) return []
  return Object.entries(raw).map(([label, stats]) => {
    const bag = isRecord(stats) ? stats : {}
    const precision =
      typeof bag.precision === 'number' || typeof bag.precision === 'string' ? bag.precision : undefined
    const recall =
      typeof bag.recall === 'number' || typeof bag.recall === 'string' ? bag.recall : undefined
    return { label, precision, recall }
  })
}

function splitN(report: ReportPayload, which: 'test' | 'validation'): number | undefined {
  if (which === 'test') {
    if (typeof report.n_test === 'number') return report.n_test
    if (typeof report.n === 'number') return report.n
  } else {
    if (typeof report.n_validation === 'number') return report.n_validation
    if (typeof report.n_val === 'number') return report.n_val
  }
  const root = isRecord(report.metrics) ? report.metrics : null
  const nested = root && isRecord(root[which]) ? root[which] : null
  if (nested && typeof nested.n === 'number') return nested.n
  const idx = root && isRecord(root.split_indices) ? root.split_indices : null
  const key = which === 'validation' ? 'val' : 'test'
  if (idx && Array.isArray(idx[key])) return idx[key].length
  return undefined
}

function metricUnit(key: string, units?: string | null): string | null {
  if (!units || units === 'none') return null
  const base = key.replace(/^(val_|test_|validation_)/, '')
  if (UNIT_METRICS.has(key) || UNIT_METRICS.has(base)) return units
  return null
}

function MetricGrid({
  rows,
  units,
}: {
  rows: Array<[string, number | string]>
  units?: string | null
}) {
  if (!rows.length) return null
  return (
    <div className="metrics">
      {rows.map(([key, value]) => {
        const unit = metricUnit(key, units)
        return (
          <div className="metric" key={key}>
            <span className="label">{humanize(key === 'score' ? 'score' : key)}</span>
            <span className="value">
              {formatNumber(value)}
              {unit ? <span className="unit"> {unit}</span> : null}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function baselineSource(report: ReportPayload): Record<string, unknown> | null {
  if (isRecord(report.baseline_metrics) && Object.keys(report.baseline_metrics).length) {
    return report.baseline_metrics
  }
  return null
}

function baselineSplitBags(raw: Record<string, unknown>): {
  test: Record<string, number | string> | null
  validation: Record<string, number | string> | null
  flat: Record<string, number | string>
} {
  const testNested = isRecord(raw.test) ? scalarBag(raw.test) : null
  const valNested =
    (isRecord(raw.validation) ? scalarBag(raw.validation) : null) ??
    (isRecord(raw.val) ? scalarBag(raw.val) : null)
  const flat = flattenMetrics(raw)
  return {
    test: testNested && Object.keys(testNested).length ? testNested : null,
    validation: valNested && Object.keys(valNested).length ? valNested : null,
    flat,
  }
}

function worseOnBag(
  model: Record<string, number | string> | null,
  baseline: Record<string, number | string> | null,
  task: string,
): boolean {
  if (!model || !baseline) return false
  const keys =
    task === 'regression' || task === 'forecast'
      ? ['mae', 'rmse', 'val_mae', 'val_rmse']
      : ['macro_f1', 'macro-f1', 'f1_macro', 'balanced_accuracy', 'val_score', 'score']
  const key = keys.find((item) => model[item] != null && baseline[item] != null)
  if (!key) return false
  const left = Number(model[key])
  const right = Number(baseline[key])
  if (!Number.isFinite(left) || !Number.isFinite(right)) return false
  if (HIGHER_BETTER.has(key)) return left < right
  return left > right
}

function worseThanBaseline(report: ReportPayload, task: string): boolean {
  const selected = String(report.selected_candidate ?? '').toLowerCase()
  if (selected === 'dummy' || selected === 'last_value' || selected === 'seasonal_naive') {
    return true
  }
  if (report.worse_than_baseline) return true
  const baseline = baselineSource(report)
  if (!baseline) return false
  const modelBags = splitBags(report)
  const baseBags = baselineSplitBags(baseline)
  if (worseOnBag(modelBags.test, baseBags.test, task)) return true
  if (worseOnBag(modelBags.validation, baseBags.validation, task)) return true
  if (!baseBags.test && !baseBags.validation) {
    return worseOnBag(modelBags.test ?? modelBags.validation, baseBags.flat, task)
  }
  return false
}

function BaselineBlock({
  report,
  task,
  units,
}: {
  report: ReportPayload
  task: string
  units?: string | null
}) {
  const baseline = baselineSource(report)
  if (!baseline) return null
  const modelBags = splitBags(report)
  const baseBags = baselineSplitBags(baseline)
  const sections: Array<{ title: string; rows: Array<[string, number | string]> }> = []
  if (baseBags.test) {
    sections.push({ title: 'Baseline · held-out test', rows: pickMetrics(baseBags.test, task) })
  }
  if (baseBags.validation) {
    sections.push({
      title: 'Baseline · validation',
      rows: pickMetrics(baseBags.validation, task, { keepScore: true }),
    })
  }
  if (!baseBags.test && !baseBags.validation && Object.keys(baseBags.flat).length) {
    sections.push({ title: 'Baseline', rows: pickMetrics(baseBags.flat, task, { keepScore: true }) })
  }
  if (!sections.some((section) => section.rows.length)) return null

  const comparisons: string[] = []
  const pushCompare = (
    label: string,
    key: string,
    model: Record<string, number | string> | null,
    base: Record<string, number | string> | null,
  ) => {
    if (!model || !base || model[key] == null || base[key] == null) return
    const unit = metricUnit(key, units)
    const unitSuffix = unit ? ` ${unit}` : ''
    comparisons.push(
      `${label}: ${formatNumber(model[key])} vs baseline ${formatNumber(base[key])}${unitSuffix}`,
    )
  }
  const compareKeys =
    task === 'regression' || task === 'forecast'
      ? ['mae', 'rmse', 'val_mae', 'val_rmse']
      : ['macro_f1', 'balanced_accuracy', 'val_score', 'score']
  for (const key of compareKeys) {
    pushCompare('Held-out test', key, modelBags.test, baseBags.test)
    pushCompare('Validation', key, modelBags.validation, baseBags.validation)
    if (!baseBags.test && !baseBags.validation) {
      pushCompare('This split', key, modelBags.test ?? modelBags.validation, baseBags.flat)
    }
  }

  return (
    <div className="baseline-block">
      {sections.map((section) => (
        <div className="metric-block" key={section.title}>
          <h4>{section.title}</h4>
          <MetricGrid rows={section.rows} units={units} />
        </div>
      ))}
      {comparisons.length ? (
        <ul className="baseline-compare">
          {comparisons.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}

function candidateNotes(report: ReportPayload): string[] {
  const notes: string[] = []
  const seen = new Set<string>()
  const push = (line: string) => {
    const trimmed = line.trim()
    if (!trimmed || seen.has(trimmed)) return
    seen.add(trimmed)
    notes.push(trimmed)
  }
  for (const warning of report.warnings ?? []) push(warning)
  const metrics = isRecord(report.metrics) ? (report.metrics as Record<string, unknown>) : null
  const failures = metrics && Array.isArray(metrics.candidate_failures) ? metrics.candidate_failures : []
  for (const fail of failures) {
    if (!isRecord(fail)) continue
    const name = String(fail.candidate ?? 'candidate')
    const err = String(fail.error ?? 'did not run')
    const friendly = /libomp|openmp/i.test(err)
      ? `${name} did not run: this computer is missing a matching OpenMP library.`
      : `${name} did not run: ${err}`
    if (![...seen].some((item) => item.toLowerCase().includes(name.toLowerCase()))) push(friendly)
  }
  const candidates = metrics && isRecord(metrics.candidates) ? metrics.candidates : null
  if (candidates) {
    for (const [name, info] of Object.entries(candidates)) {
      if (isRecord(info) && info.status === 'failed') {
        const err = String(info.error ?? 'did not run')
        if (![...seen].some((item) => item.toLowerCase().includes(name.toLowerCase()))) {
          push(`${name} did not run: ${err}`)
        }
      }
    }
  }
  return notes
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function seriesFromVectors(report: ReportPayload): SeriesMetrics[] {
  if (report.per_series?.length) return report.per_series
  const metrics = report.metrics as Record<string, unknown> | undefined
  const groups = metrics?.y_group
  const yTrue = metrics?.y_true
  const yPred = metrics?.y_pred
  if (!Array.isArray(groups) || !Array.isArray(yTrue) || !Array.isArray(yPred)) return []
  const buckets = new Map<string, { err: number; n: number }>()
  for (let i = 0; i < groups.length; i += 1) {
    const name = String(groups[i] ?? `Series ${i + 1}`)
    const t = Number(yTrue[i])
    const p = Number(yPred[i])
    if (!Number.isFinite(t) || !Number.isFinite(p)) continue
    const cur = buckets.get(name) ?? { err: 0, n: 0 }
    cur.err += Math.abs(t - p)
    cur.n += 1
    buckets.set(name, cur)
  }
  return [...buckets.entries()].map(([series, { err, n }]) => ({
    series,
    metrics: { mae: n ? err / n : 0, n },
  }))
}

function defaultLimitation(task: string): string {
  if (task === 'regression') {
    return 'These errors are typical distances from the true number, not a promise for any one row.'
  }
  if (task === 'forecast') {
    return 'The future can change for reasons that are not in this history. Treat this as a planning aid, not a guarantee.'
  }
  return 'This describes patterns in the rows you provided. It will miss rare cases and anything that was not in the spreadsheet.'
}

function asPlots(report: ReportPayload): Array<PlotRef | string> {
  const list = [...(report.plots ?? []), ...(report.plot_files ?? [])]
  if (report.confusion_matrix) list.unshift(report.confusion_matrix)
  if (report.residuals) list.unshift(report.residuals)
  return list
}

function plotTitle(plot: PlotRef | string): string {
  if (typeof plot === 'string') {
    const name = plot.split(/[\\/]/).pop() ?? plot
    if (/confusion/i.test(name)) return 'Confusion matrix'
    if (/residual/i.test(name)) return 'Residuals'
    return name.replace(/\.[a-z0-9]+$/i, '').replace(/[_-]+/g, ' ')
  }
  if (plot.title) return plot.title
  const name = plot.name ?? plot.filename ?? plot.path ?? 'Chart'
  if (/confusion/i.test(name)) return 'Confusion matrix'
  if (/residual/i.test(name)) return 'Residuals'
  return name.replace(/\.[a-z0-9]+$/i, '').replace(/[_-]+/g, ' ')
}

function PlotImage({ src, title }: { src: string; title: string }) {
  const [ok, setOk] = useState(true)
  if (!ok) return null
  return (
    <figure className="plot-wrap">
      <img className="plot" src={src} alt={title} onError={() => setOk(false)} />
      <figcaption>{title}</figcaption>
    </figure>
  )
}

export function ReportCard({ projectId, job, report, error, onCancel }: Props) {
  const status = String(job?.status ?? '')
  const active = jobActive(status)
  const progress = parseProgress(job?.progress ?? job?.progress_json)
  const percent = progress?.percent
  const message =
    progress?.message ??
    progress?.step ??
    (status === 'queued' ? 'Waiting to start…' : status === 'running' ? 'Training…' : '')

  if (status === 'succeeded' && !report) {
    if (error) {
      return (
        <p className="error" role="alert">
          {error}
        </p>
      )
    }
    return (
      <p className="caption" role="status">
        Loading results…
      </p>
    )
  }

  if (active || (job && !report && status !== 'failed' && status !== 'canceled')) {
    return (
      <div className="progress" role="status" aria-live="polite">
        <p className="caption">{message || 'Working…'}</p>
        <div className={`bar${percent == null ? ' indeterminate' : ''}`}>
          <span style={{ width: `${Math.max(4, Math.min(100, percent ?? 40))}%` }} />
        </div>
        <div className="row-actions">
          <button type="button" className="danger" onClick={onCancel}>
            Stop training
          </button>
        </div>
      </div>
    )
  }

  if (status === 'canceled') {
    return <p className="caption">Training was stopped.</p>
  }

  if (status === 'failed' || error) {
    return (
      <p className="error" role="alert">
        {error || job?.error || 'Training did not finish. You can go back and try again.'}
      </p>
    )
  }

  if (!report) {
    return <p className="caption">Train a model to see how well it did.</p>
  }

  const task = String(report.task ?? '') as TaskType | string
  const { test: testBag, validation: valBag } = splitBags(report)
  const testRows = testBag ? pickMetrics(testBag, task) : []
  const valRows = valBag ? pickMetrics(valBag, task, { keepScore: true }) : []
  const classRows = perClassRows(report)
  const plots = asPlots(report)
  const worse = worseThanBaseline(report, task)
  const seriesRows = seriesFromVectors(report)
  const testN = splitN(report, 'test')
  const valN = splitN(report, 'validation')
  const n = testN ?? report.n ?? report.n_test ?? report.n_rows
  const version = report.dataset_version ?? report.dataset_version_id
  const notes = candidateNotes(report)

  return (
    <>
      <p className="caption">
        {version != null ? `Dataset version ${version}` : 'This dataset'}
        {report.split ? ` · ${report.split} split` : ''}
        {testN != null ? ` · ${Number(testN).toLocaleString()} rows on the test split` : ''}
        {valN != null ? ` · ${Number(valN).toLocaleString()} rows on the validation split` : ''}
        {testN == null && valN == null && n != null
          ? ` · ${Number(n).toLocaleString()} rows in the check`
          : ''}
      </p>

      {testRows.length || valRows.length ? (
        <>
          {testRows.length ? (
            <div className="metric-block">
              <h4>Held-out test</h4>
              <MetricGrid rows={testRows} units={report.units} />
            </div>
          ) : null}
          {valRows.length ? (
            <div className="metric-block">
              <h4>Validation</h4>
              <MetricGrid rows={valRows} units={report.units} />
            </div>
          ) : null}
        </>
      ) : (
        <p className="caption">No summary numbers were returned for this run.</p>
      )}

      {classRows.length ? (
        <div className="preview-wrap">
          <table className="series-table class-table">
            <caption className="sr-only">Per-class precision and recall</caption>
            <thead>
              <tr>
                <th scope="col">Class</th>
                <th scope="col">Precision</th>
                <th scope="col">Recall</th>
              </tr>
            </thead>
            <tbody>
              {classRows.map((row) => (
                <tr key={row.label}>
                  <td>{row.label}</td>
                  <td className="num">{row.precision != null ? formatNumber(row.precision) : '—'}</td>
                  <td className="num">{row.recall != null ? formatNumber(row.recall) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <BaselineBlock report={report} task={task} units={report.units} />

      {worse ? (
        <p className="honest" role="status">
          This model did not beat a simple baseline. A naive guess was as good or better on the
          held-out rows.
        </p>
      ) : null}

      {plots.map((plot, index) => {
        const src = plotUrl(projectId, job?.id ?? '', plot)
        if (!src) return null
        return <PlotImage key={`${src}-${index}`} src={src} title={plotTitle(plot)} />
      })}

      {seriesRows.length ? (
        <div className="preview-wrap">
          <table className="series-table">
            <caption className="sr-only">Per-series results</caption>
            <thead>
              <tr>
                <th scope="col">Series</th>
                <th scope="col">How it did</th>
              </tr>
            </thead>
            <tbody>
              {seriesRows.map((row, index) => (
                <tr key={row.series ?? row.group ?? row.name ?? index}>
                  <td>{row.series ?? row.group ?? row.name ?? `Series ${index + 1}`}</td>
                  <td>
                    {row.metrics
                      ? Object.entries(row.metrics)
                          .map(([key, value]) => `${humanize(key)} ${formatNumber(value)}`)
                          .join(' · ')
                      : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {notes.length ? (
        <ul className="warn-list">
          {notes.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      ) : null}

      <p className="limitation">{report.limitation ?? defaultLimitation(task)}</p>
    </>
  )
}
