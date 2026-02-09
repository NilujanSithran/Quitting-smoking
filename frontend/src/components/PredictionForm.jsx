import React, { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { fetchPredict, PREDICT_FEATURES } from '../api'
import './PredictionForm.css'

const LABELS = {
  'age': 'Age',
  'waist(cm)': 'Waist (cm)',
  'fasting blood sugar': 'Fasting blood sugar',
  'triglyceride': 'Triglyceride',
  'HDL': 'HDL',
  'hemoglobin': 'Hemoglobin',
  'Gtp': 'Gtp',
}

const initialValues = {}
PREDICT_FEATURES.forEach((k) => { initialValues[k] = '' })

const MODEL_ORDER = ['Random Forest', 'SVM', 'Neural Network']

function PredictionForm() {
  const [values, setValues] = useState(initialValues)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const handleChange = (key, e) => {
    setValues((v) => ({ ...v, [key]: e.target.value }))
    setError(null)
    setResult(null)
  }

  const handleReset = () => {
    setValues({ ...initialValues })
    setResult(null)
    setError(null)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setResult(null)
    const data = {}
    for (const key of PREDICT_FEATURES) {
      const v = Number(values[key])
      if (Number.isNaN(v) || values[key] === '') {
        setError(`Please enter a number for ${LABELS[key]}.`)
        return
      }
      data[key] = v
    }
    setLoading(true)
    try {
      const res = await fetchPredict(data)
      setResult(res)
    } catch (err) {
      setError(err.message || 'Prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  const predictionsForDisplay = result
    ? MODEL_ORDER.map((name) => {
        const fromApi = (result.predictions || []).find((p) => p.model === name)
        const acc = result.model_accuracies?.[name]
        if (fromApi) return fromApi
        return { model: name, percent: null, accuracy_percent: acc }
      })
    : []

  return (
    <section className="card prediction-card">
      <h2 className="card-title">Real-time prediction</h2>
      <p className="muted">
        Insert Input Values
      </p>
      <form onSubmit={handleSubmit} className="prediction-form">
        <div className="prediction-fields">
          {PREDICT_FEATURES.map((key) => (
            <div key={key} className="field-group">
              <label htmlFor={key}>{LABELS[key]}</label>
              <input
                id={key}
                type="number"
                step="any"
                value={values[key]}
                onChange={(e) => handleChange(key, e)}
                placeholder={key}
              />
            </div>
          ))}
        </div>
        <button type="submit" className="prediction-btn" disabled={loading}>
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </form>
      {error && <p className="prediction-error">{error}</p>}
      {result && (
        <div className="prediction-results">
          <h3 className="results-title">Prediction results</h3>
          <div className="results-summary">
            <span className="summary-label">Predicted:</span>
            <span className={`summary-value ${result.predicted_class === 1 ? 'smoking' : 'nonsmoking'}`}>
              {result.predicted_label}
            </span>
            <span className="summary-percent">({result.average_percent}% smoking)</span>
          </div>
          {result.model_accuracy_percent != null && (
            <div className="results-summary">
              <span className="summary-label">Best model accuracy:</span>
              <span className="summary-value">{result.model_accuracy_percent}%</span>
              {result.model_used && <span className="muted"> ({result.model_used})</span>}
            </div>
          )}
          <div className="prediction-chart">
            <h4 className="chart-title">All results: Random Forest, SVM & Neural Network prediction %</h4>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={predictionsForDisplay.map((p) => {
                  const displayPct = p.percent != null ? p.percent : (p.accuracy_percent ?? result.model_accuracies?.[p.model] ?? 0)
                  return {
                    name: p.model,
                    percent: Number(displayPct) || 0,
                    isAccuracyOnly: p.percent == null && (p.accuracy_percent != null || result.model_accuracies?.[p.model] != null),
                  }
                })}
                margin={{ top: 8, right: 16, left: 8, bottom: 24 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="name" tick={{ fill: 'var(--muted)', fontSize: 12 }} />
                <YAxis domain={[0, 100]} tick={{ fill: 'var(--muted)', fontSize: 12 }} unit="%" />
                <Tooltip
                  contentStyle={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 8 }}
                  formatter={(value, name, props) => [
                    props.payload.isAccuracyOnly ? `${value}% (accuracy from pipeline)` : `${value}% smoking`,
                    'Prediction',
                  ]}
                  labelStyle={{ color: 'var(--text)' }}
                />
                <Bar dataKey="percent" radius={[4, 4, 0, 0]}>
                  {predictionsForDisplay.map((_, i) => (
                    <Cell key={i} fill={i === 0 ? 'var(--accent)' : i === 1 ? '#8e44ad' : '#16a085'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <ul className="technique-list">
            {predictionsForDisplay.map((p) => {
              const pct = p.percent != null ? p.percent : (p.accuracy_percent ?? result.model_accuracies?.[p.model])
              const hasAccuracy = result.model_accuracies && result.model_accuracies[p.model] != null
              return (
                <li key={p.model} className="technique-item">
                  <span className="technique-name">{p.model}</span>
                  {pct != null ? (
                    <>
                      <span className="technique-percent">{pct}%</span>
                      <span className="technique-label">{p.percent != null ? 'smoking' : '(from pipeline)'}</span>
                      {hasAccuracy && <span className="technique-accuracy">(accuracy {result.model_accuracies[p.model]}%)</span>}
                    </>
                  ) : (
                    <span className="technique-accuracy">—</span>
                  )}
                </li>
              )
            })}
          </ul>
          <button type="button" className="prediction-btn prediction-reset-btn" onClick={handleReset}>
            Reset
          </button>
        </div>
      )}
    </section>
  )
}

export default PredictionForm
