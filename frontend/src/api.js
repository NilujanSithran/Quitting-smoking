/**
 * API client for Kaggle Pipeline backend.
 * In dev (Vite): call backend directly to avoid proxy 405 on POST. In production: use /api.
 */
const API_BASE = typeof import.meta !== 'undefined' && import.meta.env?.DEV
  ? 'http://localhost:8000/api'
  : '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function fetchStats() {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error('Failed to fetch stats');
  return res.json();
}

export async function fetchResults() {
  const res = await fetch(`${API_BASE}/results`);
  if (!res.ok) throw new Error('Failed to fetch results');
  return res.json();
}

export async function fetchSubmission(limit = 100) {
  const res = await fetch(`${API_BASE}/submission?limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch submission');
  return res.json();
}

/** URL for pipeline graph images. Cache-buster so new plots show after refresh. */
export function plotUrl(filename) {
  return `${API_BASE}/plots/${filename}?t=1`;
}

/** Real-time prediction: send 7 features, get predictions from RF, SVM, NN. */
export const PREDICT_FEATURES = [
  'age',
  'waist(cm)',
  'fasting blood sugar',
  'triglyceride',
  'HDL',
  'hemoglobin',
  'Gtp',
];

export async function fetchPredict(data) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Prediction failed');
  }
  return res.json();
}
