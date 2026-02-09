import React from 'react'
import './Dashboard.css'

function Dashboard({ stats }) {
  if (!stats) {
    return (
      <section className="card dashboard">
        <h2 className="card-title">Dataset</h2>
        <p className="muted">Loading…</p>
      </section>
    )
  }

  if (stats.error) {
    return (
      <section className="card dashboard">
        <h2 className="card-title">Dataset</h2>
        <p className="muted">{stats.error}</p>
      </section>
    )
  }

  const dist = stats.target_distribution || {}
  const nonSmoker = dist[0] ?? 0
  const smoker = dist[1] ?? 0
  const total = nonSmoker + smoker
  const pctSmoker = total ? ((smoker / total) * 100).toFixed(1) : '—'

  return (
    <section className="card dashboard">
      <h2 className="card-title">Dataset</h2>
      <div className="dashboard-grid">
        <div className="stat">
          <span className="stat-value">{stats.rows.toLocaleString()}</span>
          <span className="stat-label">Rows</span>
        </div>
        <div className="stat">
          <span className="stat-value">{stats.columns}</span>
          <span className="stat-label">Columns</span>
        </div>
        <div className="stat">
          <span className="stat-value">{pctSmoker}%</span>
          <span className="stat-label">Smoker (class 1)</span>
        </div>
        <div className="stat">
          <span className="stat-value">{nonSmoker.toLocaleString()} / {smoker.toLocaleString()}</span>
          <span className="stat-label">Non-smoker / Smoker</span>
        </div>
      </div>
    </section>
  )
}

export default Dashboard
