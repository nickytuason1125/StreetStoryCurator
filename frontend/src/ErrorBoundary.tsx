import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props { children: ReactNode }
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
    const { error, info } = this.state
    if (!error) return this.props.children

    return (
      <div style={{
        position: 'fixed', inset: 0, overflow: 'auto', zIndex: 99999,
        background: '#0b0e14', color: '#e5e7eb',
        font: '13px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace',
        padding: '32px 40px',
      }}>
        <h2 style={{ color: '#ff6b6b', fontSize: 18, margin: '0 0 8px' }}>
          Something broke while rendering
        </h2>
        <p style={{ color: '#9ca3af', margin: '0 0 20px' }}>
          The app hit an error and stopped drawing this screen. Your graded photos are safe —
          try “Back to gallery”, or reload if that doesn’t help.
        </p>
        <div style={{ display: 'flex', gap: 10, marginBottom: 24 }}>
          <button onClick={this.reset}
            style={{ padding: '8px 16px', fontSize: 13, fontWeight: 600, cursor: 'pointer',
              background: '#3b82f6', color: '#fff', border: 'none', borderRadius: 8 }}>
            Back to gallery
          </button>
          <button onClick={() => window.location.reload()}
            style={{ padding: '8px 16px', fontSize: 13, fontWeight: 600, cursor: 'pointer',
              background: 'transparent', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 8 }}>
            Reload app
          </button>
        </div>
        <div style={{ color: '#ff8a8a', fontWeight: 600, marginBottom: 6 }}>
          {String(error.message ?? error)}
        </div>
        <pre style={{ whiteSpace: 'pre-wrap', color: '#9ca3af', margin: 0 }}>
          {error.stack ?? ''}
          {info?.componentStack ?? ''}
        </pre>
      </div>
    )
  }
}
