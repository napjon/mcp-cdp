import { useRef, useState } from 'react'
import { parseProgress } from '../../api'
import type { Job } from '../../types'
import { jobActive } from '../../workflow'

type Props = {
  ready: boolean
  busy: boolean
  error: string | null
  job: Job | null
  predictionId: string | null
  onScore: (file: File) => void
  onDownload: () => void
  onCancel?: () => void
}

export function PredictCard({
  ready,
  busy,
  error,
  job,
  predictionId,
  onScore,
  onDownload,
  onCancel,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [active, setActive] = useState(false)
  const status = String(job?.status ?? '')
  const progress = parseProgress(job?.progress ?? job?.progress_json)
  const working = busy || jobActive(status)
  const percent = progress?.percent
  const message =
    progress?.message ??
    progress?.step ??
    (status === 'queued' ? 'Waiting to start…' : status === 'running' ? 'Scoring rows…' : 'Scoring rows…')

  if (working) {
    return (
      <div className="progress" role="status" aria-live="polite">
        <p className="caption">{message}</p>
        <div className={`bar${percent == null ? ' indeterminate' : ''}`}>
          <span style={{ width: `${Math.max(4, Math.min(100, percent ?? 40))}%` }} />
        </div>
        {onCancel ? (
          <div className="row-actions">
            <button type="button" className="danger" onClick={onCancel}>
              Stop scoring
            </button>
          </div>
        ) : null}
      </div>
    )
  }

  if (!ready) {
    return <p className="caption">Train a model first, then you can score a new spreadsheet.</p>
  }

  return (
    <>
      <p className="caption">
        Upload another CSV with the same kinds of columns. You will get a scored file back.
      </p>
      <div
        className={`dropzone${active ? ' active' : ''}`}
        onDragEnter={(event) => {
          event.preventDefault()
          setActive(true)
        }}
        onDragOver={(event) => {
          event.preventDefault()
          setActive(true)
        }}
        onDragLeave={() => setActive(false)}
        onDrop={(event) => {
          event.preventDefault()
          setActive(false)
          const file = event.dataTransfer.files[0]
          if (file) onScore(file)
        }}
      >
        <strong>Drop a new CSV to score</strong>
        <span>Same columns as the file you trained on work best.</span>
        <div>
          <input
            ref={fileRef}
            className="sr-only"
            type="file"
            accept=".csv,text/csv"
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) onScore(file)
              event.target.value = ''
            }}
          />
          <button
            type="button"
            className="primary file-btn"
            disabled={working}
            onClick={() => fileRef.current?.click()}
          >
            Choose CSV
          </button>
        </div>
      </div>

      {error ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}

      {status === 'failed' && job?.error ? (
        <p className="error" role="alert">
          {job.error}
        </p>
      ) : null}

      {predictionId ? (
        <div className="row-actions">
          <button type="button" className="primary" onClick={onDownload}>
            Download scored file
          </button>
        </div>
      ) : null}
    </>
  )
}
