import { useEffect, useRef } from 'react'

type Props = {
  value: string
  sending: boolean
  disabled?: boolean
  onChange: (value: string) => void
  onSend: () => void
  onStop: () => void
}

export function Composer({ value, sending, disabled, onChange, onSend, onStop }: Props) {
  const ref = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [value])

  return (
    <div className="composer-dock">
      <form
        className="composer"
        id="composer"
        onSubmit={(event) => {
          event.preventDefault()
          if (sending) onStop()
          else onSend()
        }}
      >
        <label className="sr-only" htmlFor="composer-input">
          Message
        </label>
        <textarea
          id="composer-input"
          ref={ref}
          rows={1}
          value={value}
          disabled={disabled}
          placeholder="Ask a question, or start by adding a spreadsheet above."
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.nativeEvent.isComposing) return
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              if (!sending) onSend()
            }
          }}
        />
        {sending ? (
          <button type="button" className="danger" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="submit" className="primary" disabled={disabled || !value.trim()}>
            Send
          </button>
        )}
      </form>
      <p className="composer-hint">Enter to send · Shift+Enter for a new line</p>
    </div>
  )
}
