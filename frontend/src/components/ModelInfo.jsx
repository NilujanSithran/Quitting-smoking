import React from 'react'
import './ModelInfo.css'

const TECHNIQUES = ['Random Forest', 'SVM', 'Neural Network']

function ModelInfo({ results }) {
  const info = results && results.model_info
  const models = (results?.models || []).filter((m) => TECHNIQUES.includes(m.name))
  const bestModel = models.length ? models.reduce((a, b) => (b.accuracy > a.accuracy ? b : a), models[0]) : null
  const bestName = bestModel?.name ?? results?.best_model
  const bestAcc = bestModel?.accuracy ?? results?.best_accuracy

  if (!bestName && !info) {
    return (
      <section className="card model-info-card">
        <h2 className="card-title">Best model</h2>
        <p className="muted">Run pipeline.py to train and save the best model.</p>
      </section>
    )
  }

  const sizeStr = info && (info.size_mb != null ? `${info.size_mb} MB` : info.size_bytes != null ? `${(info.size_bytes / 1024).toFixed(1)} KB` : null)

  return (
    <section className="card model-info-card">
      <h2 className="card-title">Best model - {bestName || info?.name}</h2>
      <div className="model-info-grid">
        <div className="model-info-item">
          <span className="model-info-label">Best model</span>
          <span className="model-info-value">{bestName || info?.name}</span>
        </div>
        <div className="model-info-item">
          <span className="model-info-label">Accuracy</span>
          <span className="model-info-value">{bestAcc != null ? `${(bestAcc * 100).toFixed(2)}%` : '—'}</span>
        </div>
        {info && (
          <div className="model-info-item">
            <span className="model-info-label">File</span>
            <span className="mono">{info.path}</span>
          </div>
        )}
        {sizeStr && (
          <div className="model-info-item">
            <span className="model-info-label">Size</span>
            <span className="mono">{sizeStr}</span>
          </div>
        )}
      </div>
    </section>
  )
}

export default ModelInfo
