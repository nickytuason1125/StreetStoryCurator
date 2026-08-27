import { useEffect, useMemo, useRef, useState } from 'react';
import { CornerDownLeft, Search } from 'lucide-react';
import { cn } from '../../lib/cn';
import { T } from '../../theme/tokens';
import { Kbd, KbdHint } from './Kbd';

/* CommandPalette — the ⌘K surface, and the app's primary control surface.
 *
 * Cockpit pass. This was a 288px dropdown borrowed from the side panel's width,
 * with 24px rows and a hover tint for selection. It is now sized and weighted
 * like the thing you are meant to reach for first: a 560px sheet, a real query
 * field rather than a chip, 36px rows, and a selection state you cannot lose
 * track of while typing.
 *
 * Three deliberate choices:
 *
 * 1. The active row is marked by a 2px --ai bar in the gutter, NOT by colour
 *    alone. Both states reserve the gutter, so the label never shifts by two
 *    pixels as the selection moves — in a list you drive at typing speed, that
 *    jitter is what makes a palette feel cheap.
 * 2. Groups are headers, not separators. When results filter down to two rows
 *    the headings vanish with them, so the list never shows an empty section.
 * 3. Every row prints its shortcut. The palette's job is partly to TEACH the
 *    keyboard: a command you reach twice by mouse should be one you then know
 *    by key.
 *
 * The machine voice (--ai) marks the palette's own chrome — its selection bar
 * and its enter glyph — because the palette is AI-era furniture, not a
 * photographer judgement. It never appears on the caps: those are chrome.
 */

export type PaletteAction = {
  id: string;
  label: string;
  hint?: string;
  kbd?: string;
  /** Section heading. Actions with no group fall into a trailing "Other". */
  group?: string;
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
    return actions.filter(a => {
      const t = a.label.toLowerCase();
      let i = 0;
      for (const ch of q) {
        i = t.indexOf(ch, i);
        if (i === -1) return false;
        i += 1;
      }
      return true;
    });
  }, [actions, query]);

  /* Rows carry their flat index so arrow-key maths stays independent of how
     many headings sit between them. */
  const rows = useMemo(() => {
    const out: Array<{ heading: string } | { action: PaletteAction; i: number }> = [];
    let seen: string | null = null;
    hits.forEach((a, i) => {
      const g = a.group ?? 'Other';
      if (g !== seen) { out.push({ heading: g }); seen = g; }
      out.push({ action: a, i });
    });
    return out;
  }, [hits]);

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
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[14vh]"
      style={{ background: T.scrim }}
      onMouseDown={() => setOpen(false)}
    >
      <div
        role="dialog" aria-label="Command palette"
        className="glass elev-3 animate-palette-in flex w-palette flex-col overflow-hidden rounded-md border border-line-strong"
        onMouseDown={e => e.stopPropagation()}
      >
        {/* Query field. A real input at --h-field: the palette is typed at, so
            the field is the largest thing in it. */}
        <div className="flex h-field shrink-0 items-center gap-3 border-b border-line px-4">
          <Search size={16} className="shrink-0 text-ink-3"/>
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
            placeholder="Search commands…"
            aria-label="Command search"
            /* The global :focus-visible ring is suppressed HERE and only here.
               It out-specifies Tailwind's `outline-none` (both are one class,
               and index.css lands after the utilities layer), which painted a
               browser-default box around the query field. The dialog itself is
               the focus context — the field is the only thing in it that can
               take a caret, so a ring adds nothing and reads as a defect. */
            className="w-full border-0 bg-transparent text-md text-ink outline-none
                       focus-visible:outline-none focus-visible:shadow-none placeholder:text-ink-4"
          />
          <Kbd>esc</Kbd>
        </div>

        <div ref={listRef} className="max-h-palette overflow-y-auto py-2">
          {hits.length === 0 && (
            <div className="px-4 py-3 text-sm text-ink-3">
              No command matches <span className="text-ink-2">{query}</span>
            </div>
          )}

          {rows.map(row => (
            'heading' in row ? (
              <div key={`h-${row.heading}`} className="t-label px-4 pb-1 pt-2">
                {row.heading}
              </div>
            ) : (
              <button
                key={row.action.id}
                data-active={row.i === idx ? '1' : '0'}
                onMouseEnter={() => setIdx(row.i)}
                onMouseDown={e => e.preventDefault()}
                onClick={() => commit(row.action)}
                className={cn(
                  'flex h-row w-full items-center gap-3 border-0 border-l-2 px-4 text-left',
                  'transition-colors duration-fast ease',
                  row.i === idx
                    ? 'border-l-ai bg-ai-dim'
                    : 'border-l-transparent bg-transparent hover:bg-raised',
                )}
              >
                <span className={cn('flex-1 truncate text-sm', row.i === idx ? 'text-ink' : 'text-ink-2')}>
                  {row.action.label}
                </span>
                {row.action.hint && (
                  <span className="shrink-0 truncate text-xs text-ink-3">{row.action.hint}</span>
                )}
                {row.action.kbd && <Kbd>{row.action.kbd}</Kbd>}
                {row.i === idx
                  ? <CornerDownLeft size={13} className="shrink-0" style={{ color: T.ai }}/>
                  : <span className="w-3 shrink-0"/>}
              </button>
            )
          ))}
        </div>

        {/* Footer legend. The palette teaches its own controls. */}
        <div className="flex h-8 shrink-0 items-center gap-4 border-t border-line bg-surface px-4">
          <KbdHint keys="↑↓" label="Navigate"/>
          <KbdHint keys="↵" label="Run"/>
          <KbdHint keys="esc" label="Close"/>
          <span className="flex-1"/>
          <span className="text-xs text-ink-4">
            <span className="t-num">{hits.length}</span> of <span className="t-num">{actions.length}</span>
          </span>
        </div>
      </div>
    </div>
  );
}
