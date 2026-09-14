import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { DatasetDetail, Job, ReportPayload } from '../src/types.ts'
import { initialState, reducer, type AppState } from '../src/workflow.ts'

const dataset: DatasetDetail = {
  id: 'ds1',
  name: 'sales.csv',
  source_type: 'csv',
}

const columns = [{ name: 'y', inferred_role: 'numeric' as const }]

const trainJob: Job = { id: 'train-1', type: 'train', status: 'succeeded' }
const predictJob: Job = { id: 'pred-1', type: 'predict', status: 'succeeded' }
const report: ReportPayload = { task: 'regression', metrics: { mae: 1.2 }, units: 'kg' }

function trained(): AppState {
  let state = initialState()
  state = reducer(state, {
    type: 'BOOT',
    project: { id: 'p1', name: 'Local workspace' },
    appStatus: null,
    apiReachable: true,
  })
  state = reducer(state, { type: 'DATA_LOADED', dataset, preview: null, columns })
  state = reducer(state, { type: 'SET_TASK', task: 'regression' })
  state = reducer(state, { type: 'SET_TARGET', target: 'y' })
  state = reducer(state, { type: 'CONFIRM_TARGET' })
  state = reducer(state, { type: 'CONTINUE_COLUMNS' })
  state = reducer(state, { type: 'TRAIN_QUEUED', experimentId: 'exp-1', job: trainJob })
  state = reducer(state, { type: 'REPORT_LOADED', report, modelId: 'model-1', jobId: 'train-1' })
  state = reducer(state, { type: 'PREDICT_QUEUED', job: predictJob })
  state = reducer(state, { type: 'PREDICT_READY', predictionId: 'out-1', jobId: 'pred-1' })
  return state
}

function hasCard(state: AppState, card: string): boolean {
  return state.thread.some((item) => item.kind === 'card' && item.card === card)
}

function assertArtifactsCleared(state: AppState) {
  assert.equal(state.trainJob, null)
  assert.equal(state.report, null)
  assert.equal(state.modelId, null)
  assert.equal(state.predictionId, null)
  assert.equal(state.predictJob, null)
}

test('SET_SPLIT clears downstream artifacts and keeps Confirm', () => {
  const state = reducer(trained(), { type: 'SET_SPLIT', split: 'group' })
  assertArtifactsCleared(state)
  assert.equal(state.split, 'group')
  assert.equal(hasCard(state, 'confirm'), true)
  assert.equal(hasCard(state, 'report'), false)
})

test('SET_BUDGET, SET_FORECAST, SET_ROLE, SET_TASK, SET_TARGET clear artifacts', () => {
  const budget = reducer(trained(), { type: 'SET_BUDGET', budget: 'thorough' })
  assertArtifactsCleared(budget)

  const forecast = reducer(trained(), { type: 'SET_FORECAST', horizon: 21 })
  assertArtifactsCleared(forecast)

  const role = reducer(trained(), { type: 'SET_ROLE', name: 'y', role: 'excluded' })
  assertArtifactsCleared(role)

  const task = reducer(trained(), { type: 'SET_TASK', task: 'classification' })
  assertArtifactsCleared(task)

  const target = reducer(trained(), { type: 'SET_TARGET', target: 'y2' })
  assertArtifactsCleared(target)
})

test('SET_ALLOW_EXCLUDE clears artifacts but keeps the checkbox and Confirm card', () => {
  const state = reducer(trained(), { type: 'SET_ALLOW_EXCLUDE', value: true })
  assertArtifactsCleared(state)
  assert.equal(state.allowExcludeInsufficient, true)
  assert.equal(hasCard(state, 'confirm'), true)
})

test('stale REPORT_LOADED and JOB_UPDATE cannot repopulate a reset run', () => {
  const cleared = reducer(trained(), { type: 'SET_SPLIT', split: 'time' })
  const afterReport = reducer(cleared, {
    type: 'REPORT_LOADED',
    report,
    modelId: 'model-1',
    jobId: 'train-1',
  })
  assert.equal(afterReport.report, null)
  assert.equal(afterReport.modelId, null)

  const afterJob = reducer(cleared, { type: 'JOB_UPDATE', kind: 'train', job: trainJob })
  assert.equal(afterJob.trainJob, null)
})

test('RESTORE_RUN hydrates a succeeded train job and report', () => {
  let state = initialState()
  state = reducer(state, {
    type: 'BOOT',
    project: { id: 'p1', name: 'Local workspace' },
    appStatus: null,
    apiReachable: true,
  })
  state = reducer(state, {
    type: 'RESTORE_RUN',
    experimentId: 'exp-1',
    trainJob,
    report,
    modelId: 'model-1',
    predictionId: 'out-1',
  })
  assert.equal(state.trainJob?.id, 'train-1')
  assert.equal(state.report?.units, 'kg')
  assert.equal(state.modelId, 'model-1')
  assert.equal(state.predictionId, 'out-1')
  assert.equal(hasCard(state, 'report'), true)
  assert.equal(hasCard(state, 'predict'), true)
})
