import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

const rootEl = document.getElementById('root')

const showBootError = (message) => {
  if (!rootEl) return
  rootEl.innerHTML = `
    <div style="min-height:100vh;display:flex;align-items:center;justify-content:center;background:#0b1220;color:#e2e8f0;font-family:'Space Grotesk',sans-serif;padding:24px;text-align:center;">
      <div>
        <h1 style="font-size:22px;margin-bottom:8px;">UI 오류 발생</h1>
        <p style="color:#94a3b8;line-height:1.6;">${message || '화면을 불러오는 중 문제가 발생했습니다.'}</p>
      </div>
    </div>
  `
}

window.addEventListener('error', (event) => {
  const payload = {
    type: 'error',
    message: event?.message || 'Unknown error',
    source: event?.filename,
    lineno: event?.lineno,
    colno: event?.colno,
    stack: event?.error?.stack
  }
  fetch('/client_error', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).catch(() => {})
  showBootError(payload.message)
})

window.addEventListener('unhandledrejection', (event) => {
  const payload = {
    type: 'unhandledrejection',
    message: event?.reason?.message || String(event?.reason || 'Promise rejection'),
    stack: event?.reason?.stack
  }
  fetch('/client_error', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).catch(() => {})
  showBootError(payload.message)
})

createRoot(rootEl).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
