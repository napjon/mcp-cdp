import { useEffect, useId, useRef, useState } from 'react'
import {
  isGidUnspecified,
  isNotFound,
  plainError,
  previewDisclosure,
  previewSheet,
  putDisclosure,
  SHEET_TAB_HELP,
  sheetTabGid,
} from '../../api'
import type { DatasetDetail, PreviewTable } from '../../types'

type Props = {
  projectId: string
  dataset: DatasetDetail | null
  preview: PreviewTable | null
  busy: boolean
  error: string | null
  onUpload: (file: File) => void
  onSheet: (url: string, headerRow: number) => void
  onRefresh?: () => void
}

function previewCaption(shown: number, total: number | undefined): string {
  if (total != null && shown < total) {
    return `Showing ${shown.toLocaleString()} of ${total.toLocaleString()}`
  }
  if (total != null) {
    return `Showing all ${total.toLocaleString()} rows.`
  }
  return `Showing ${shown.toLocaleString()} preview row${shown === 1 ? '' : 's'}.`
}

function SampleTable({
  preview,
  label,
  total,
}: {
  preview: PreviewTable
  label: string
  total?: number
}) {
  const previewTotal = total ?? preview.total
  return (
    <>
      <div className="preview-wrap" tabIndex={0} aria-label={label}>
        <table>
          <caption className="sr-only">{previewCaption(preview.shown, previewTotal)}</caption>
          <thead>
            <tr>
              {preview.columns.map((col) => (
                <th key={col} scope="col">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="caption">{previewCaption(preview.shown, previewTotal)}</p>
    </>
  )
}

export function DataCard({
  projectId,
  dataset,
  preview,
  busy,
  error,
  onUpload,
  onSheet,
  onRefresh,
}: Props) {
  const inputId = useId()
  const fileRef = useRef<HTMLInputElement>(null)
  const [active, setActive] = useState(false)
  const [url, setUrl] = useState('')
  const [headerDraft, setHeaderDraft] = useState<{ datasetId: string | null; value: number } | null>(
    null,
  )
  const [tabHelp, setTabHelp] = useState(false)
  const [sheetBusy, setSheetBusy] = useState(false)
  const [sheetError, setSheetError] = useState<string | null>(null)
  const [pendingSheet, setPendingSheet] = useState<{
    url: string
    headerRow: number
    preview: PreviewTable
  } | null>(null)
  useEffect(() => {
    setPendingSheet(null)
    setSheetError(null)
  }, [dataset?.id])

  const storedHeader = dataset?.current_version?.header_row ?? dataset?.version?.header_row
  const headerRow =
    headerDraft && headerDraft.datasetId === (dataset?.id ?? null)
      ? headerDraft.value
      : typeof storedHeader === 'number'
        ? storedHeader
        : 0

  function takeFile(file: File | undefined) {
    if (!file) return
    onUpload(file)
  }

  async function onPreviewSheet() {
    const trimmed = url.trim()
    if (!trimmed) return
    if (!sheetTabGid(trimmed)) {
      setTabHelp(true)
      setPendingSheet(null)
      return
    }
    if (!projectId) {
      setSheetError('The local app is not reachable yet.')
      return
    }
    setTabHelp(false)
    setSheetError(null)
    const row = Number.isFinite(headerRow) ? Math.max(0, Math.floor(headerRow)) : 0
    setSheetBusy(true)
    try {
      const sample = await previewSheet(projectId, {
        url: trimmed,
        header_row: row,
        gid: sheetTabGid(trimmed) ?? undefined,
      })
      setPendingSheet({ url: trimmed, headerRow: row, preview: sample })
    } catch (err) {
      setPendingSheet(null)
      if (isNotFound(err)) {
        setSheetError(
          'Sheet preview is not available on this server, so this link was not connected. Nothing was saved.',
        )
      } else {
        setSheetError(plainError(err))
      }
    } finally {
      setSheetBusy(false)
    }
  }

  const nRows = dataset?.current_version?.n_rows ?? dataset?.profile?.n_rows ?? preview?.total
  const nCols = dataset?.current_version?.n_cols ?? dataset?.profile?.n_cols ?? preview?.columns.length
  const previewTotal = preview?.total ?? nRows
  const showTabHelp =
    tabHelp || (error != null && (error === SHEET_TAB_HELP || isGidUnspecified('', error)))
  const formBusy = busy || sheetBusy

  return (
    <>
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
          takeFile(event.dataTransfer.files[0])
        }}
      >
        <strong>Drop a CSV here</strong>
        <span>or choose a file from this computer.</span>
        <div>
          <input
            id={inputId}
            ref={fileRef}
            className="sr-only"
            type="file"
            accept=".csv,text/csv"
            onChange={(event) => {
              takeFile(event.target.files?.[0])
              event.target.value = ''
            }}
          />
          <button
            type="button"
            className="primary file-btn"
            disabled={formBusy}
            onClick={() => fileRef.current?.click()}
          >
            Choose CSV
          </button>
        </div>
      </div>

      <form
        className="field"
        onSubmit={(event) => {
          event.preventDefault()
          void onPreviewSheet()
        }}
      >
        <label htmlFor="sheets-url">
          Or paste a Google Sheets link
          <span className="hint"> — share it with “Anyone with the link”.</span>
        </label>
        <input
          id="sheets-url"
          type="url"
          inputMode="url"
          placeholder="https://docs.google.com/spreadsheets/d/…#gid="
          value={url}
          disabled={formBusy}
          onChange={(event) => {
            setUrl(event.target.value)
            if (tabHelp) setTabHelp(false)
            setPendingSheet(null)
            setSheetError(null)
          }}
        />
        <label htmlFor="sheets-header-row">
          Header row
          <span className="hint"> — 0 is the first row. Applied to the preview.</span>
        </label>
        <input
          id="sheets-header-row"
          type="number"
          min={0}
          step={1}
          value={headerRow}
          disabled={formBusy}
          onChange={(event) => {
            const value = event.target.value
            setHeaderDraft({
              datasetId: dataset?.id ?? null,
              value: value === '' ? 0 : Number(value),
            })
            setPendingSheet(null)
          }}
        />
        <div className="row-actions">
          <button type="submit" className="ghost" disabled={formBusy || !url.trim()}>
            Use this sheet
          </button>
          {pendingSheet ? (
            <button
              type="button"
              className="primary"
              disabled={formBusy}
              onClick={() => onSheet(pendingSheet.url, pendingSheet.headerRow)}
            >
              Connect this sheet
            </button>
          ) : null}
          {dataset?.source_type === 'sheets' && dataset.connection_id && onRefresh ? (
            <button type="button" className="ghost" disabled={formBusy} onClick={onRefresh}>
              Refresh sheet
            </button>
          ) : null}
        </div>
      </form>

      {sheetBusy ? (
        <p className="caption" role="status">
          Loading a preview of this sheet…
        </p>
      ) : null}
      {busy ? (
        <p className="caption" role="status">
          Reading the spreadsheet…
        </p>
      ) : null}
      {sheetError ? (
        <p className="error" role="alert">
          {sheetError}
        </p>
      ) : null}
      {error ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}
      {showTabHelp && error !== SHEET_TAB_HELP && sheetError !== SHEET_TAB_HELP ? (
        <p className="error" role="alert">
          {SHEET_TAB_HELP}
        </p>
      ) : null}

      {pendingSheet ? (
        <div className="sheet-pending">
          <p className="caption">
            Preview with header row {pendingSheet.headerRow}. Confirm to connect this sheet. Nothing
            is saved until you confirm.
          </p>
          {pendingSheet.preview.rows.length > 0 ? (
            <SampleTable preview={pendingSheet.preview} label="Google Sheet preview" />
          ) : (
            <p className="caption">The preview did not include any rows.</p>
          )}
        </div>
      ) : null}

      {dataset ? (
        <p className="caption">
          Loaded <strong>{dataset.name}</strong>
          {nRows != null ? ` · ${nRows.toLocaleString()} rows` : ''}
          {nCols != null ? ` · ${nCols.toLocaleString()} columns` : ''}
          {dataset.source_type === 'sheets' ? ' · Google Sheet' : ' · CSV'}
        </p>
      ) : null}

      {!pendingSheet && preview && preview.rows.length > 0 ? (
        <SampleTable
          preview={preview}
          label="Spreadsheet preview"
          total={previewTotal ?? undefined}
        />
      ) : null}

      {dataset ? (
        <DisclosurePanel projectId={projectId} datasetId={dataset.id} />
      ) : null}
    </>
  )
}

function DisclosurePanel({ projectId, datasetId }: { projectId: string; datasetId: string }) {
  const [hidden, setHidden] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [enabled, setEnabled] = useState(false)
  const [sample, setSample] = useState<PreviewTable | null>(null)
  const [previewed, setPreviewed] = useState(false)

  useEffect(() => {
    setHidden(false)
    setError(null)
    setEnabled(false)
    setSample(null)
    setPreviewed(false)
  }, [projectId, datasetId])

  if (hidden || !projectId || !datasetId) return null

  async function onPreview() {
    setBusy(true)
    setError(null)
    try {
      const result = await previewDisclosure(projectId, datasetId)
      setSample(result.preview)
      setPreviewed(true)
      setEnabled(result.disclosure.enabled)
    } catch (err) {
      if (isNotFound(err)) {
        setHidden(true)
        return
      }
      setError(plainError(err))
    } finally {
      setBusy(false)
    }
  }

  async function onSetEnabled(next: boolean) {
    setBusy(true)
    setError(null)
    try {
      const result = await putDisclosure(projectId, datasetId, next)
      if (result.error && result.ok === false) {
        setError(result.error)
        return
      }
      setEnabled(result.enabled)
    } catch (err) {
      if (isNotFound(err)) {
        setHidden(true)
        return
      }
      setError(plainError(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="disclosure">
      <p className="caption">
        Chat does not see raw rows unless you allow it. Preview the exact sample first, then enable
        or disable sharing.
      </p>
      <div className="row-actions">
        <button type="button" className="ghost" disabled={busy} onClick={() => void onPreview()}>
          Preview sample
        </button>
        {previewed ? (
          <>
            <button
              type="button"
              className="ghost"
              disabled={busy || enabled}
              onClick={() => void onSetEnabled(true)}
            >
              Enable
            </button>
            <button
              type="button"
              className="ghost"
              disabled={busy || !enabled}
              onClick={() => void onSetEnabled(false)}
            >
              Disable
            </button>
          </>
        ) : null}
      </div>
      {busy ? (
        <p className="caption" role="status">
          {previewed ? 'Updating sample sharing…' : 'Loading the exact sample…'}
        </p>
      ) : null}
      {error ? (
        <p className="error" role="alert">
          {error}
        </p>
      ) : null}
      {previewed && enabled ? (
        <p className="caption">Sample sharing is on for this spreadsheet.</p>
      ) : null}
      {previewed && !enabled ? (
        <p className="caption">Sample sharing is off. Chat will not receive these rows.</p>
      ) : null}
      {sample && sample.rows.length > 0 ? (
        <SampleTable preview={sample} label="Sample that chat would receive" />
      ) : null}
    </div>
  )
}
