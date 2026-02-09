import React from 'react'
import './TrainingFeatures.css'

function TrainingFeatures({ results }) {
  const cols = results?.feature_cols
  if (!cols || !Array.isArray(cols) || cols.length === 0) {
    return null
  }

  return (
    <section className="card training-features-card">
      <h2 className="card-title">Features used for training</h2>
      <p className="muted">
        These columns from the training dataset are used to train the model (selected by importance, target: <code className="mono">smoking</code>).
      </p>
      <ul className="feature-cols-list">
        {cols.map((name) => (
          <li key={name} className="feature-col-item">
            <code className="mono">{name}</code>
          </li>
        ))}
      </ul>
      <p className="muted feature-count">{cols.length} features</p>
    </section>
  )
}

export default TrainingFeatures
