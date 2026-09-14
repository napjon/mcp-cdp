import assert from 'node:assert/strict'
import { test } from 'node:test'
import { previewDisclosure, readSse, type SseEventResult } from '../src/api.ts'

function responseFrom(chunks: string[]): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
  return new Response(stream, { headers: { 'content-type': 'text/event-stream' } })
}

function onEvent(data: unknown): SseEventResult {
  if (data === '[DONE]') return 'done'
  if (data && typeof data === 'object' && 'type' in data) {
    const type = String((data as { type: unknown }).type).toLowerCase()
    if (type === 'done' || type === 'complete' || type === 'finished' || type === 'end') {
      return 'done'
    }
    if (type === 'interrupted' || type === 'abort') return 'interrupted'
    if (type === 'error' || type === 'failed') return 'error'
  }
  return 'continue'
}

test('readSse EOF without a done event is interrupted', async () => {
  const outcome = await readSse(responseFrom(['data: {"type":"delta","text":"hi"}\n\n']), onEvent)
  assert.equal(outcome, 'interrupted')
})

test('readSse with no body is interrupted', async () => {
  const outcome = await readSse(new Response(null), onEvent)
  assert.equal(outcome, 'interrupted')
})

test('readSse sets done only when onEvent returns done', async () => {
  const outcome = await readSse(responseFrom(['data: {"type":"done"}\n\n']), onEvent)
  assert.equal(outcome, 'done')
})

test('readSse honors a terminal event in the trailing buffer', async () => {
  const outcome = await readSse(responseFrom(['data: {"type":"failed"}']), onEvent)
  assert.equal(outcome, 'error')
})

test('readSse keeps trailing non-terminal data interrupted', async () => {
  const outcome = await readSse(responseFrom(['data: {"type":"delta","text":"partial"}']), onEvent)
  assert.equal(outcome, 'interrupted')
})

test('previewDisclosure keeps an omitted sharing status unknown', async (t) => {
  const originalFetch = globalThis.fetch
  t.after(() => {
    globalThis.fetch = originalFetch
  })
  globalThis.fetch = (async () =>
    new Response(JSON.stringify({ columns: ['name'], rows: [['Ada']], previewed_at: 'now' }), {
      headers: { 'content-type': 'application/json' },
    })) as typeof fetch

  const result = await previewDisclosure('project-1', 'dataset-1')
  assert.equal(result.disclosure.enabled, null)
  assert.deepEqual(result.preview.rows, [['Ada']])
})
