import type { ReactNode } from 'react'

type Props = {
  step: number
  title: string
  summary?: string
  expanded: boolean
  complete?: boolean
  onToggle: () => void
  children: ReactNode
}

export function CardFrame({
  step,
  title,
  summary,
  expanded,
  complete,
  onToggle,
  children,
}: Props) {
  const headingId = `card-${step}-title`
  return (
    <section className="card" aria-labelledby={headingId}>
      <div className="card-head">
        <div>
          <span className="card-kicker">
            Step {step}
            {complete ? ' · done' : ''}
          </span>
          <h3 id={headingId}>{title}</h3>
          {!expanded && summary ? <p className="card-summary">{summary}</p> : null}
        </div>
        <button type="button" className="ghost" onClick={onToggle}>
          {expanded ? 'Hide' : complete ? 'Change' : 'Open'}
        </button>
      </div>
      {expanded ? <div className="card-body">{children}</div> : null}
    </section>
  )
}
