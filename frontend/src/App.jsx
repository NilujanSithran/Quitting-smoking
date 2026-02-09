import React, { useState, useEffect } from 'react'
import { fetchHealth, fetchStats, fetchResults, fetchSubmission } from './api'
import Dashboard from './components/Dashboard.jsx'
import ModelComparison from './components/ModelComparison.jsx'
import ModelInfo from './components/ModelInfo.jsx'
import TrainingFeatures from './components/TrainingFeatures.jsx'
import PredictionForm from './components/PredictionForm.jsx'
import Plots from './components/Plots.jsx'
import SubmissionTable from './components/SubmissionTable.jsx'
import Header from './components/Header.jsx'
import './App.css'

function App() {
  const [apiStatus, setApiStatus] = useState('loading')
  const [stats, setStats] = useState(null)
  const [results, setResults] = useState(null)
  const [submission, setSubmission] = useState(null)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        await fetchHealth()
        if (cancelled) return
        setApiStatus('ok')
        const [s, r, sub] = await Promise.all([
          fetchStats(),
          fetchResults(),
          fetchSubmission(100),
        ])
        if (!cancelled) {
          setStats(s)
          setResults(r)
          setSubmission(sub)
        }
      } catch (e) {
        if (!cancelled) setApiStatus('error')
      }
    }
    load()
    return () => { cancelled = true }
  }, [])

  return (
    <div className="app">
      <Header apiStatus={apiStatus} />
      <main className="main">
        <Dashboard stats={stats} />
        <ModelComparison results={results} />
        <ModelInfo results={results} />
        {/* <TrainingFeatures results={results} /> */}
        <PredictionForm />
        <Plots results={results} />
        <SubmissionTable submission={submission} />
      </main>
    </div>
  )
}

export default App
