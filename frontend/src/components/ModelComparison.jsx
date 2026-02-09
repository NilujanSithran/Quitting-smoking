import React from 'react'
import './ModelComparison.css'

// Only these three techniques are used and displayed; nothing else is included
const TECHNIQUES = ['Random Forest', 'SVM', 'Neural Network']

function ModelComparison({ results }) {
  if (!results) {
    return (
      <section className="card model-comparison">
        <h2 className="card-title">Model comparison</h2>
        <p className="muted">Loading…</p>
      </section>
    )
  }

  const allModels = results.models || []
  const filtered = allModels.filter((m) => TECHNIQUES.includes(m.name))
  const models = [...filtered].sort((a, b) => (b.accuracy - a.accuracy))

  if (!results.available || models.length === 0) {
    return (
      <section className="card model-comparison">
        <h2 className="card-title">Model comparison</h2>
        <p className="muted">{results.message || 'Run the pipeline (pipeline.py) to generate results for Random Forest, SVM, and Neural Network.'}</p>
      </section>
    )
  }

  // Best = model with highest accuracy (always from displayed data so badge matches the numbers)
  const bestModel = models.reduce((a, b) => (b.accuracy > a.accuracy ? b : a), models[0])
  const best = bestModel?.name ?? null
  const bestAccuracy = bestModel?.accuracy ?? null

  return (
    <section className="card model-comparison">
      <div className="model-comparison-head">
        <h2 className="card-title">Random Forest, SVM & Neural Network</h2>
        {best && (
          <span className="best-badge">
            Best: {best}
            {bestAccuracy != null && ` (${(bestAccuracy * 100).toFixed(2)}%)`}
          </span>
        )}
      </div>
      <div className="models-grid">
        {models.map((m) => (
          <div
            key={m.name}
            className={`model-card ${m.name === best ? 'model-card-best' : ''}`}
          >
            <h3 className="model-name">{m.name}</h3>
            <div className="model-metrics">
              <div className="metric">
                <span className="metric-value">{(m.accuracy * 100).toFixed(2)}%</span>
                <span className="metric-label">Accuracy</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

export default ModelComparison
