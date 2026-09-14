import { COLUMN_ROLES, ROLE_LABEL, type ColumnProfile, type ColumnRole } from '../../types'
import { excludeReason, isConstantColumn } from '../../workflow'

type Props = {
  columns: ColumnProfile[]
  roles: Record<string, ColumnRole>
  target: string | null
  dateColumn: string | null
  onRole: (name: string, role: ColumnRole) => void
  onContinue: () => void
  error?: string | null
}

export function ColumnsCard({
  columns,
  roles,
  target,
  dateColumn,
  onRole,
  onContinue,
  error,
}: Props) {
  return (
    <>
      <p className="caption">
        We guessed what each column is. Click the type to change it. IDs and columns that never
        change are left out unless you say otherwise.
      </p>
      <div className="badge-list">
        {columns.map((col) => {
          const role = roles[col.name] ?? col.inferred_role
          const locked = col.name === target
          const reason = role === 'excluded' ? excludeReason(col) : null
          return (
            <div key={col.name} className={`badge${role === 'excluded' ? ' excluded' : ''}`}>
              <span>
                {col.name}
                {locked ? ' (target)' : ''}
                {col.name === dateColumn ? ' (date)' : ''}
              </span>
              {locked ? (
                <span className={`role-tag ${role}`}>{ROLE_LABEL[role]}</span>
              ) : (
                <label className={`role-tag ${role} role-menu`}>
                  <span className="sr-only">Type for {col.name}</span>
                  <select
                    value={role}
                    onChange={(event) => onRole(col.name, event.target.value as ColumnRole)}
                  >
                    {COLUMN_ROLES.map((option) => (
                      <option key={option} value={option}>
                        {ROLE_LABEL[option]}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {reason ? (
                <span className="sr-only">{reason}</span>
              ) : isConstantColumn(col) && role !== 'excluded' ? (
                <span className="sr-only">This column was constant; you chose to include it.</span>
              ) : null}
            </div>
          )
        })}
      </div>
      <ul className="col-note" style={{ paddingLeft: 18, margin: 0 }}>
        {columns
          .filter((col) => (roles[col.name] ?? col.inferred_role) === 'excluded' && excludeReason(col))
          .map((col) => (
            <li key={col.name}>
              <strong>{col.name}:</strong> {excludeReason(col)}
            </li>
          ))}
      </ul>
      {error ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}
      <div className="row-actions">
        <button type="button" className="primary" onClick={onContinue}>
          Use these columns
        </button>
      </div>
    </>
  )
}
