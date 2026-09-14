import type { ThreadMessage } from '../types'

export function MessageBubble({ item }: { item: ThreadMessage }) {
  const streaming = item.status === 'streaming'
  return (
    <div
      className={`bubble ${item.role}${item.status === 'failed' ? ' failed' : ''}`}
      aria-live={item.role === 'assistant' ? 'polite' : undefined}
    >
      {item.content || (streaming ? '' : item.status === 'failed' ? 'The reply did not finish.' : '')}
      {streaming ? <span className="cursor" aria-hidden="true" /> : null}
      {item.status === 'interrupted' ? <span className="meta">Stopped</span> : null}
      {item.status === 'failed' && item.content ? (
        <span className="meta">This reply did not finish cleanly.</span>
      ) : null}
    </div>
  )
}
