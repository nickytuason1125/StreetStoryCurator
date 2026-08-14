import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { flushSync } from 'react-dom'
// Bundled locally, never a CDN — the app must run fully offline (contract rule 5).
// The `wdth` build carries BOTH axes (weight 100-900, stretch 62-125%), which is
// what lets labels come from Archivo's width axis instead of a third family.
import '@fontsource-variable/archivo/wdth.css'
import '@fontsource-variable/martian-mono/wdth.css'
import './theme/tokens.css'
import './index.css'
import App from './App.tsx'
import ErrorBoundary from './ErrorBoundary.tsx'

try {
  const root = createRoot(document.getElementById('root')!)
  flushSync(() => {
    root.render(
      <StrictMode>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </StrictMode>,
    )
  })
} catch (err) {
  document.body.style.cssText = 'margin:0;background:#0e0e13;color:#ff6b6b;font-family:monospace;padding:20px'
  document.body.innerHTML = '<h2>Render error</h2><pre>' + String(err) + '</pre>'
}
