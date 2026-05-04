import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { flushSync } from 'react-dom'
import './index.css'
import App from './App.tsx'

try {
  const root = createRoot(document.getElementById('root')!)
  flushSync(() => {
    root.render(
      <StrictMode>
        <App />
      </StrictMode>,
    )
  })
} catch (err) {
  document.body.style.cssText = 'margin:0;background:#0e0e13;color:#ff6b6b;font-family:monospace;padding:20px'
  document.body.innerHTML = '<h2>Render error</h2><pre>' + String(err) + '</pre>'
}
