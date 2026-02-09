import React from 'react'
import './Header.css'

function Header({ apiStatus }) {
  return (
    <header className="header">
      <div className="header-inner">
        <h1 className="header-title">Quitting smoking</h1>
        <p className="header-subtitle">Model training & comparison</p>
        <div className="header-status">
          <span className={`status-dot status-${apiStatus}`} />
          <span className="status-label">
            {apiStatus === 'ok' && 'API connected'}
            {apiStatus === 'loading' && 'Connecting…'}
            {apiStatus === 'error' && 'API offline — start backend (port 8000)'}
          </span>
        </div>
      </div>
    </header>
  )
}

export default Header
