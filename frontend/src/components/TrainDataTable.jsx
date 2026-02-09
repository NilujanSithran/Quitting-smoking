import React from 'react'
import './TrainDataTable.css'

function TrainDataTable({ trainData }) {
  if (!trainData) {
    return (
      <section className="card train-data-card">
        <h2 className="card-title">Training data</h2>
        <p className="muted">Loading…</p>
      </section>
    )
  }

  if (!trainData.available || !trainData.rows || trainData.rows.length === 0) {
    return (
      <section className="card train-data-card">
        <h2 className="card-title">Training data</h2>
        <p className="muted">No training data available.</p>
      </section>
    )
  }

  const columns = Object.keys(trainData.rows[0] || {})

  return (
    <section className="card train-data-card">
      <div className="train-data-head">
        <h2 className="card-title">Training data</h2>
        <span className="train-data-total">
          Showing {trainData.rows.length} of {trainData.total.toLocaleString()} rows
        </span>
      </div>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {trainData.rows.map((row, i) => (
              <tr key={row.id ?? i}>
                {columns.map((col) => (
                  <td key={col} className="mono">
                    {String(row[col] ?? '')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export default TrainDataTable
