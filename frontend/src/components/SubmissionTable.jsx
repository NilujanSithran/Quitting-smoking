import React from 'react'
import './SubmissionTable.css'

function SubmissionTable({ submission }) {
  if (!submission) {
    return (
      <section className="card submission-card">
        <h2 className="card-title">Submission preview</h2>
        <p className="muted">Loading…</p>
      </section>
    )
  }

  if (!submission.available || !submission.rows || submission.rows.length === 0) {
    return (
      <section className="card submission-card">
        <h2 className="card-title">Submission preview</h2>
        <p className="muted">No submission file yet.</p>
        <p className="muted steps-hint">From the project folder run <code className="mono">python pipeline.py</code> (with <code className="mono">test.csv</code> present). Then refresh this page. Ensure the backend is running: <code className="mono">python app.py</code>.</p>
      </section>
    )
  }

  return (
    <section className="card submission-card">
      <div className="submission-head">
        <h2 className="card-title">Submission preview</h2>
        <span className="submission-total">{submission.total.toLocaleString()} rows</span>
      </div>
      <div className="table-wrap">
        <table className="submission-table">
          <thead>
            <tr>
              <th>id</th>
              <th>smoking</th>
            </tr>
          </thead>
          <tbody>
            {submission.rows.map((row, i) => (
              <tr key={row.id ?? i}>
                <td className="mono">{row.id}</td>
                <td className="mono">{row.smoking}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export default SubmissionTable
