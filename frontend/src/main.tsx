import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { flushSync } from 'react-dom'
// Bundled locally, never a CDN — the app must run fully offline (contract rule 5).
// Three voices, each with one job: Space Grotesk carries the display/brand
// voice (technical, engineered — the 2026 register), Geist carries the UI
// (tight, tall x-height, legible down to 11px labels), and Geist Mono every
// number (tabular figures, slashed zero via .t-num).
import '@fontsource-variable/space-grotesk'
import '@fontsource-variable/geist'
import '@fontsource-variable/geist-mono'
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
  // Safe to use tokens here: tokens.css is imported above, so the custom
  // properties are already applied even though React never mounted.
  document.body.style.cssText =
    'margin:0;background:var(--well);color:var(--alarm-crit);font-family:var(--font-mono);padding:var(--sp-4)'
  // Built via DOM APIs with textContent — never innerHTML. Error text can
  // contain fragments of arbitrary page data; injecting it as markup would be
  // an injection sink even in a local-only app.
  const h = document.createElement('h2')
  h.textContent = 'Render error'
  const pre = document.createElement('pre')
  pre.textContent = String(err)
  document.body.replaceChildren(h, pre)
}
