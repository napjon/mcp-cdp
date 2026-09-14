import type { ColumnProfile, DuplicateTimestamp, Frequency, TaskType } from '../../types'
import { TASK_GOAL } from '../../types'

type Props = {
  columns: ColumnProfile[]
  task: TaskType | null
  target: string | null
  dateColumn: string | null
  groupColumns: string[]
  frequency: Frequency
  horizon: number
  duplicateTimestamp: DuplicateTimestamp
  confirmed: boolean
  onTask: (task: TaskType) => void
  onTarget: (target: string) => void
  onForecast: (patch: {
    dateColumn?: string | null
    groupColumns?: string[]
    frequency?: Frequency
    horizon?: number
    duplicateTimestamp?: DuplicateTimestamp
  }) => void
  onConfirm: () => void
  seriesOverLimit?: boolean
  seriesNote?: string | null
}

const TASKS: TaskType[] = ['classification', 'regression', 'forecast']

export function TargetCard({
  columns,
  task,
  target,
  dateColumn,
  groupColumns,
  frequency,
  horizon,
  duplicateTimestamp,
  confirmed,
  onTask,
  onTarget,
  onForecast,
  onConfirm,
  seriesOverLimit = false,
  seriesNote = null,
}: Props) {
  const canConfirm =
    Boolean(task && target) &&
    (task !== 'forecast' || Boolean(dateColumn)) &&
    horizon >= 1 &&
    horizon <= 90 &&
    !seriesOverLimit

  return (
    <>
      <p className="caption">What should we try to predict? Pick a column, then choose the kind of answer you want.</p>
      <label className="field">
        <span>Column to predict</span>
        <select
          value={target ?? ''}
          onChange={(event) => onTarget(event.target.value)}
          aria-required="true"
        >
          <option value="" disabled>
            Choose a column
          </option>
          {columns.map((col) => (
            <option key={col.name} value={col.name}>
              {col.name}
            </option>
          ))}
        </select>
      </label>

      <div className="goal-grid" role="radiogroup" aria-label="Kind of prediction">
        {TASKS.map((key) => {
          const goal = TASK_GOAL[key]
          const selected = task === key
          return (
            <button
              key={key}
              type="button"
              className={`goal${selected ? ' selected' : ''}`}
              role="radio"
              aria-checked={selected}
              onClick={() => onTask(key)}
            >
              <strong>{goal.title}</strong>
              <span>{goal.blurb}</span>
              <em>Example: {goal.example}</em>
            </button>
          )
        })}
      </div>

      {task === 'forecast' ? (
        <>
          <label className="field">
            <span>Date column</span>
            <select
              value={dateColumn ?? ''}
              onChange={(event) => onForecast({ dateColumn: event.target.value || null })}
            >
              <option value="">Choose the date column</option>
              {columns.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>
              Groups <span className="hint">(optional — product, store, …)</span>
            </span>
            <select
              multiple
              value={groupColumns}
              onChange={(event) => {
                const selected = Array.from(event.target.selectedOptions).map((opt) => opt.value)
                onForecast({ groupColumns: selected })
              }}
              size={Math.min(6, Math.max(3, columns.length))}
            >
              {columns
                .filter((col) => col.name !== target && col.name !== dateColumn)
                .map((col) => (
                  <option key={col.name} value={col.name}>
                    {col.name}
                  </option>
                ))}
            </select>
            <span className="hint">Hold Command or Ctrl to pick more than one.</span>
          </label>
          <div className="row-actions">
            <label className="field" style={{ flex: 1 }}>
              <span>How often</span>
              <select
                value={frequency}
                onChange={(event) => onForecast({ frequency: event.target.value as Frequency })}
              >
                <option value="D">Every day</option>
                <option value="W">Every week</option>
                <option value="M">Every month</option>
              </select>
            </label>
            <label className="field" style={{ flex: 1 }}>
              <span>How far ahead</span>
              <input
                type="number"
                min={1}
                max={90}
                value={horizon}
                onChange={(event) => onForecast({ horizon: Number(event.target.value) })}
              />
            </label>
          </div>
          {seriesOverLimit ? (
            <p className="error" role="alert">
              {seriesNote || 'There are too many separate series to forecast. The limit is 100.'}
            </p>
          ) : seriesNote ? (
            <p className="caption">{seriesNote}</p>
          ) : null}
          <label className="field">
            <span>If two rows share the same date</span>
            <select
              value={duplicateTimestamp}
              onChange={(event) =>
                onForecast({ duplicateTimestamp: event.target.value as DuplicateTimestamp })
              }
            >
              <option value="reject">Stop and tell me</option>
              <option value="aggregate_mean">Average them</option>
            </select>
          </label>
        </>
      ) : null}

      <div className="row-actions">
        <button type="button" className="primary" disabled={!canConfirm} onClick={onConfirm}>
          Yes, predict this
        </button>
        {confirmed ? <span className="caption">Saved. You can still change it.</span> : null}
      </div>
    </>
  )
}
