import { useEffect, useMemo, useRef, useState } from 'react';
import { Command, CornerDownLeft, Search } from 'lucide-react';
import { cn } from '../../lib/cn';
import { T } from '../../theme/tokens';

/* CommandPalette — the ⌘K surface.
 *
 * 2026 UX pass: every global action the app can do becomes a typeahead row,
 * so the keyboard is a complete control surface (the status-bar shortcut
 * strip stays as the discoverable subset). The palette registers its own
 * Ctrl/⌘-K listener, so App only has to mount it and hand it actions.
 *
 * Visual contract: glass elevation over the scrim, rows enter on the spring,
 * the machine voice (--ai) marks the palette's own chrome (its icon and the
 * active-row indicator) because the palette is AI-era furniture, not a
 * photographer judgement.
 */

export type PaletteAction = {
  id: string;
  label: string;
  hint?: string;
  kbd?: string;
  run: () => void;
};

export function CommandPalette({ actions }: { actions: PaletteAction[] }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [idx, setIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  /* Register ⌘K / Ctrl-K once. */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setOpen(o => !o);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  /* Focus the field and reset state whenever the palette opens. */
  useEffect(() => {
    if (open) {
      setQuery('');
      setIdx(0);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  /* Case-insensitive subsequence match: "gd" finds "Grade folder". */
  const hits = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return actions;
    let i = 0;
    return actions.filter(a => {
      const t = a.label.toLowerCase();
      for (const ch of q) {
        i = t.indexOf(ch, i);
        if (i === -1) return false;
        i += 1;
      }
      return true;
    });
  }, [actions, query]);

  useEffect(() => { setIdx(0); }, [query]);

  /* Arrow keys must never move focus off the input, so the list follows the
     selection by scrolling the active row into view instead. */
  useEffect(() => {
    listRef.current?.querySelector('[data-active="1"]')
      ?.scrollIntoView({ block: 'nearest' });
  }, [idx, hits]);

  if (!open) return null;

  const commit = (a?: PaletteAction) => {
    if (!a) return;
    setOpen(false);
    // Run after the palette unmounts, so a command that opens a modal isn't
    // fighting this one for focus.
    requestAnimationFrame(() => a.run());
  };

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[18vh]"
      style={{ background: T.scrim }}
      onMouseDown={() => setOpen(false)}
    >
      <div
        role="dialog" aria-label="Command palette"
        className="glass elev-3 animate-palette-in w-panel flex flex-col overflow-hidden rounded-lg border border-line-strong"
        onMouseDown={e => e.stopPropagation()}
      >
        <div className="flex h-8 shrink-0 items-center gap-2 border-b border-line px-3">
          <Command size={12} style={{ color: T.ai }}/>
          <input
            ref={inputRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'ArrowDown') { e.preventDefault(); setIdx(i => Math.min(i + 1, hits.length - 1)); }
              else if (e.key === 'ArrowUp') { e.preventDefault(); setIdx(i => Math.max(i - 1, 0)); }
              else if (e.key === 'Enter') { e.preventDefault(); commit(hits[idx]); }
              else if (e.key === 'Escape') { e.preventDefault(); setOpen(false); }
            }}
            placeholder="Type a command…"
            aria-label="Command search"
            className="w-full border-0 bg-transparent text-sm text-ink outline-none placeholder:text-ink-4"
          />
          <kbd className="t-num rounded-sm border border-line-strong bg-raised px-1 text-xs text-ink-3">esc</kbd>
        </div>

        <div ref={listRef} className="max-h-palette overflow-y-auto py-1">
          {hits.length === 0 && (
            <div className="px-3 py-2 text-sm text-ink-3">No matching command</div>
          )}
          {hits.map((a, i) => (
            <button
              key={a.id}
              data-active={i === idx ? '1' : '0'}
              onMouseEnter={() => setIdx(i)}
              onMouseDown={e => e.preventDefault()}
              onClick={() => commit(a)}
              className={cn(
                'flex w-full items-center gap-2 border-0 px-3 py-1 text-left transition-colors duration-fast ease',
                i === idx ? 'bg-ai-dim' : 'bg-transparent',
              )}
            >
              {i === idx
                ? <CornerDownLeft size={11} className="shrink-0" style={{ color: T.ai }}/>
                : <span className="w-[11px] shrink-0"/>}
              <span className={cn('flex-1 truncate text-sm', i === idx ? 'text-ink' : 'text-ink-2')}>
                {a.label}
              </span>
              {a.hint && <span className="text-xs text-ink-3">{a.hint}</span>}
              {a.kbd && (
                <kbd className="t-num rounded-sm border border-line-strong bg-raised px-1 text-xs text-ink-3">{a.kbd}</kbd>
              )}
            </button>
          ))}
        </div>

        <div className="flex h-6 shrink-0 items-center gap-2 border-t border-line px-3">
          <Search size={9} className="shrink-0 text-ink-4"/>
          <span className="text-xs text-ink-4">
            <span className="t-num">{hits.length}</span> of <span className="t-num">{actions.length}</span> commands
          </span>
          <span className="flex-1"/>
          <span className="t-label !text-ink-4">⌘K</span>
        </div>
      </div>
    </div>
  );
}
