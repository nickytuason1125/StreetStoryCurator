import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props { children: ReactNode; /** inline = contain within a panel (per-view); overlay = fullscreen (root). */ variant?: 'overlay' | 'inline'; label?: string }
interface State { error: Error | null; info: ErrorInfo | null }

/**
 * Catches render-time exceptions anywhere in the tree.
 *
 * Without this, an uncaught error during render unmounts the whole React tree
 * and leaves the bare dark <body> — the "blank screen after grading" symptom.
 * Here we show the error + stack instead, and stash it on window.__errors so the
 * pywebview launcher diagnostic (local_launcher._diag) can log it to crash.log.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, info: null }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    this.setState({ info })
    try {
      const w = window as any
      w.__errors = w.__errors || []
      w.__errors.push({
        message: String(error?.message ?? error),
        stack: error?.stack ?? '',
        componentStack: info?.componentStack ?? '',
      })
    } catch { /* ignore */ }
    // Keep it in the devtools console too.
    console.error('[ErrorBoundary]', error, info)
  }

  reset = () => this.setState({ error: null, info: null })

  render() {
    const { error, info, variant = 'overlay', label } = this.props
    if (!error) return this.props.children

    // Per-view containment: one broken panel must not blank the whole app.
    // The inline variant renders in-flow where the view was — the rest of the
    // chrome (header, status bar, tabs) keeps working and stays clickable.
    if (variant === 'inline') {
      return (
        <div style={{
          flex: 1, minHeight: 0, overflow: 'auto',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          gap: 'var(--sp-2)', background: 'var(--ground)',
          border: '1px solid var(--line-strong)', borderRadius: 'var(--r-md)',
          margin: 'var(--sp-2)', padding: 'var(--sp-4)',
        }}>
          <span className="t-label" style={{ color: 'var(--alarm-crit)' }}>
            {label ?? 'This view'} failed to render
          </span>
          <span style={{ color: 'var(--ink-3)', fontSize: 'var(--text-sm)', textAlign: 'center' }}>
            {String(error.message ?? error).slice(0, 200)}
          </span>
          <button onClick={this.reset}
            style={{ padding: 'var(--sp-1) var(--sp-3)', fontSize: 'var(--text-xs)', fontWeight: 600, cursor: 'pointer',
              background: 'var(--raised)', color: 'var(--ink)', border: '1px solid var(--line-strong)', borderRadius: 'var(--r-sm)' }}>
            Retry
          </button>
        </div>
      )
    }

    return (
      // Tokens are safe here even though the tree failed to render: tokens.css
      // is imported in main.tsx before React mounts, so the custom properties
      // are on :root regardless of what threw.
      <div style={{
        position: 'fixed', inset: 0, overflow: 'auto', zIndex: 99999,
        background: 'var(--well)', color: 'var(--ink)',
        font: 'var(--text-sm)/1.5 var(--font-mono)',
        padding: 'var(--sp-8) var(--sp-12)',
      }}>
        <h2 style={{ color: 'var(--alarm-crit)', fontSize: 'var(--text-md)', margin: '0 0 var(--sp-2)' }}>
          Something broke while rendering
        </h2>
        <p style={{ color: 'var(--ink-2)', margin: '0 0 var(--sp-4)' }}>
          The app hit an error and stopped drawing this screen. Your graded photos are safe —
          try “Back to gallery”, or reload if that doesn’t help.
        </p>
        <div style={{ display: 'flex', gap: 'var(--sp-2)', marginBottom: 'var(--sp-6)' }}>
          {/* Emphasis by luminance, not hue — the recovery action is not a
              destructive one, and the accent is reserved regardless. */}
          <button onClick={this.reset}
            style={{ padding: 'var(--sp-2) var(--sp-4)', fontSize: 'var(--text-sm)', fontWeight: 600, cursor: 'pointer',
              background: 'var(--raised)', color: 'var(--ink)', border: '1px solid var(--line-strong)', borderRadius: 'var(--r-sm)' }}>
            Back to gallery
          </button>
          <button onClick={() => window.location.reload()}
            style={{ padding: 'var(--sp-2) var(--sp-4)', fontSize: 'var(--text-sm)', fontWeight: 600, cursor: 'pointer',
              background: 'transparent', color: 'var(--ink-2)', border: '1px solid var(--line-strong)', borderRadius: 'var(--r-sm)' }}>
            Reload app
          </button>
        </div>
        <div style={{ color: 'var(--alarm-crit)', fontWeight: 600, marginBottom: 'var(--sp-1)' }}>
          {String(error.message ?? error)}
        </div>
        <pre style={{ whiteSpace: 'pre-wrap', color: 'var(--ink-2)', margin: 0 }}>
          {error.stack ?? ''}
          {info?.componentStack ?? ''}
        </pre>
      </div>
    )
  }
}
