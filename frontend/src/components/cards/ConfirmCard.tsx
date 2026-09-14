import type {
  Budget,
  ColumnProfile,
  ExperimentConfig,
  SplitType,
  TaskType,
} from '../../types'
import { TASK_GOAL } from '../../types'
import {
  HORIZON_MAX,
  HORIZON_MIN,
  insufficientHistoryGroups,
  mentionsInsufficientHistory,
} from '../../workflow'

type Props = {
  config: ExperimentConfig | null
  task: TaskType | null
  columns: ColumnProfile[]
  warnings: string[]
  seriesNote: string | null
  seriesOverLimit: boolean
  busy: boolean
  error: string | null
  allowExcludeInsufficient: boolean
  onAllowExclude: (value: boolean) => void
  onSplit: (split: SplitType) => void
  onBudget: (budget: Budget) => void
  onDateColumn: (name: string | null) => void
  onGroupColumns: (names: string[]) => void
  onTrain: () => void
}

const SPLIT_COPY: Record<SplitType, string> = {
  random: 'Hold out a random 20% of rows to check the model.',
  group: 'Keep whole groups together so we don’t peek at the same store or product.',
  time: 'Train on earlier dates, then check on later ones.',
}

export function ConfirmCard({
  config,
  task,
  columns,
  warnings,
  seriesNote,
  seriesOverLimit,
  busy,
  error,
  allowExcludeInsufficient,
  onAllowExclude,
  onSplit,
  onBudget,
  onDateColumn,
  onGroupColumns,
  onTrain,
}: Props) {
  if (!config || !task) {
    return <p className="caption">Finish the earlier steps first.</p>
  }
  const horizon = config.horizon
  const horizonBad =
    task === 'forecast' &&
    (horizon == null || horizon < HORIZON_MIN || horizon > HORIZON_MAX)
  const historyTexts = [...warnings, error]
  const historyGroups = insufficientHistoryGroups(historyTexts)
  const showExclude = mentionsInsufficientHistory(historyTexts) || historyGroups.length > 0
  const listedWarnings = warnings.filter((warning) => !/insufficient history/i.test(warning))
  const blocked =
    busy ||
    (task !== 'forecast' && !config.features.length) ||
    (config.split === 'group' && config.group_columns.length === 0) ||
    ((task === 'forecast' || config.split === 'time') && !config.date_column) ||
    horizonBad ||
    seriesOverLimit
  const otherColumns = columns.filter((col) => col.name !== config.target)
  return (
    <>
      <div className="review">
        <dl>
          <dt>Predict</dt>
          <dd>
            {config.target}{' '}
            <span className="hint">({TASK_GOAL[task].title.toLowerCase()})</span>
          </dd>
          <dt>Using</dt>
          <dd>{config.features.length ? config.features.join(', ') : 'No input columns yet'}</dd>
          <dt>Left out</dt>
          <dd>{config.excluded.length ? config.excluded.join(', ') : 'Nothing extra'}</dd>
          {task === 'forecast' ? (
            <>
              <dt>Time</dt>
              <dd>
                {config.date_column ?? '—'} ·{' '}
                {config.frequency === 'D' ? 'daily' : config.frequency === 'W' ? 'weekly' : 'monthly'} ·{' '}
                {config.horizon} ahead
              </dd>
              <dt>Groups</dt>
              <dd>{config.group_columns.length ? config.group_columns.join(', ') : 'None'}</dd>
            </>
          ) : null}
        </dl>
      </div>

      <fieldset className="field" style={{ border: 0, padding: 0 }}>
        <legend className="label">How to check the model</legend>
        <div className="seg">
          {(['random', 'group', 'time'] as SplitType[]).map((split) => (
            <button
              key={split}
              type="button"
              aria-pressed={config.split === split}
              onClick={() => onSplit(split)}
            >
              {split === 'random' ? 'Random rows' : split === 'group' ? 'Keep groups together' : 'Earlier vs later'}
            </button>
          ))}
        </div>
        <p className="hint">{SPLIT_COPY[config.split]}</p>
        {config.split === 'time' || task === 'forecast' ? (
          <label className="field">
            <span>Date column</span>
            <select
              value={config.date_column ?? ''}
              onChange={(event) => onDateColumn(event.target.value || null)}
            >
              <option value="">Choose the date column</option>
              {otherColumns.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        {config.split === 'group' || task === 'forecast' ? (
          <label className="field">
            <span>Group columns</span>
            <select
              multiple
              value={config.group_columns}
              onChange={(event) => {
                onGroupColumns(Array.from(event.target.selectedOptions).map((opt) => opt.value))
              }}
              size={Math.min(5, Math.max(3, otherColumns.length))}
            >
              {otherColumns
                .filter((col) => col.name !== config.date_column)
                .map((col) => (
                  <option key={col.name} value={col.name}>
                    {col.name}
                  </option>
                ))}
            </select>
            <span className="hint">Hold Command or Ctrl to pick more than one.</span>
          </label>
        ) : null}
      </fieldset>

      <fieldset className="field" style={{ border: 0, padding: 0 }}>
        <legend className="label">How long to try</legend>
        <div className="seg">
          <button type="button" aria-pressed={config.budget === 'quick'} onClick={() => onBudget('quick')}>
            Quick
          </button>
          <button type="button" aria-pressed={config.budget === 'thorough'} onClick={() => onBudget('thorough')}>
            Thorough
          </button>
        </div>
        <p className="hint">
          {config.budget === 'quick'
            ? 'Tries a few solid methods. Usually a few minutes.'
            : 'Spends more time searching. Use this if Quick is not good enough.'}
        </p>
      </fieldset>

      {listedWarnings.length ? (
        <ul className="warn-list">
          {listedWarnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      ) : null}

      {seriesNote && !seriesOverLimit ? <p className="caption">{seriesNote}</p> : null}

      {showExclude ? (
        <div className="exclude-block">
          <p className="caption">These series do not have enough history:</p>
          {historyGroups.length ? (
            <ul className="warn-list">
              {historyGroups.map((group) => (
                <li key={group}>{group}</li>
              ))}
            </ul>
          ) : (
            <ul className="warn-list">
              {historyTexts
                .filter((text): text is string => !!text && /insufficient history/i.test(text))
                .filter((text, index, all) => all.indexOf(text) === index)
                .map((text) => (
                  <li key={text}>{text}</li>
                ))}
            </ul>
          )}
          <label className="check">
            <input
              type="checkbox"
              checked={allowExcludeInsufficient}
              onChange={(event) => onAllowExclude(event.target.checked)}
            />
            <span>Leave out series that do not have enough history.</span>
          </label>
        </div>
      ) : null}

      {error && !/insufficient history/i.test(error) ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}

      <div className="row-actions">
        <button
          type="button"
          className="primary"
          disabled={blocked}
          onClick={onTrain}
        >
          {busy ? 'Starting…' : 'Train'}
        </button>
      </div>
    </>
  )
}
