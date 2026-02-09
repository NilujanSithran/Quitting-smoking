import React, { useState } from 'react'
import { plotUrl } from '../api'
import './Plots.css'

const DISPLAY_ORDER = [
  'roc_curves.png',
  'confusion_matrices.png',
  'eda.png',
  'model_comparison.png',
  'target_count.png',
  'histogram_numeric.png',
  'box_plot.png',
  'correlation_heatmap.png',
  'bar_categorical_target.png',
  'pair_plot.png',
]
const KNOWN_LABELS = {
  'roc_curves.png': 'ROC Curves – Model Comparison',
  'confusion_matrices.png': 'Confusion Matrices (Validation Set)',
  'eda.png': 'EDA – Training Data',
  'model_comparison.png': 'Model Performance Comparison',
  'target_count.png': 'Target Variable Count Plot',
  'histogram_numeric.png': 'Histogram (Numeric Features)',
  'box_plot.png': 'Box Plot',
  'correlation_heatmap.png': 'Correlation Heatmap',
  'bar_categorical_target.png': 'Bar Chart (Categorical vs Target)',
  'pair_plot.png': 'Pair Plot (Top 4–5 Important Features)',
}

/** Turn filename like "target_count.png" into "Target count" for display. */
function labelForFilename(filename) {
  if (KNOWN_LABELS[filename]) return KNOWN_LABELS[filename]
  const base = filename.replace(/\.(png|jpg|jpeg)$/i, '')
  return base.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function Plots({ results }) {
  const [failed, setFailed] = useState(new Set())
  const rawFiles = Array.isArray(results?.plot_files) ? results.plot_files : []
  const files = [...DISPLAY_ORDER.filter((f) => rawFiles.includes(f)), ...rawFiles.filter((f) => !DISPLAY_ORDER.includes(f))]

  return (
    <section className="card plots-card">
      <h2 className="card-title">Graphs</h2>
      {results == null ? (
        <p className="muted">Loading…</p>
      ) : files.length === 0 ? (
        <div className="plots-empty">
          <p className="muted">No graphs yet.</p>
          <p className="muted steps-hint">From the project folder run <code className="mono">python pipeline.py</code>. It will create <code className="mono">submission.csv</code> and the <code className="mono">plots/</code> folder. Then refresh this page. Ensure the backend is running: <code className="mono">python app.py</code>.</p>
        </div>
      ) : (
        <>
          <p className="plots-hint">Model comparison, confusion matrices, ROC curves, EDA, and exploratory plots.</p>
          <div className="plots-grid">
            {files.map((filename) => (
              <figure key={filename} className="plot-figure">
                {failed.has(filename) ? (
                  <div className="plot-placeholder">
                    Failed to load image. Check <code className="mono">plots/{filename}</code>.
                  </div>
                ) : (
                  <img
                    src={plotUrl(filename)}
                    alt={labelForFilename(filename)}
                    className="plot-img"
                    onError={() => setFailed((s) => new Set(s).add(filename))}
                  />
                )}
                <figcaption className="plot-caption">{labelForFilename(filename)}</figcaption>
              </figure>
            ))}
          </div>
        </>
      )}
    </section>
  )
}

export default Plots
