import { memo, useEffect, useRef, useState } from 'react';
import { CheckSquare, Copy, Flag, Layers, RefreshCw } from 'lucide-react';
import { Button } from '../ui/Button';
import { AnnotatedMark } from '../ui/GradeRule';
import { T, gradeRule, gradeBadge, gradeKey, gradeLabel, formatScore } from '../../theme/tokens';
import { cn } from '../../lib/cn';
import { Thumb } from '../photo/Thumb';
import { useWindowedGrid } from '../../hooks/useWindowedGrid';

/* Filmstrip frame. Extracted verbatim from App.tsx during the views split —
 * zero behaviour change, one file per view. */
export const FilmThumb = memo(function FilmThumb({
  p, isSel, onSelect, isUsed, isSelected, h = 84, showFn = true, w,
}: { p: any; isSel: boolean; onSelect: (id: string) => void; isUsed: boolean; isSelected: boolean; h?: number; showFn?: boolean; w?: number }) {
  // Frames keep their real proportions here too — a fixed row height with
  // natural width, so a vertical reads as a vertical while scrubbing.
  // `w` pins the cell to a fixed 3:2 slot (object-contain letterboxes the
  // well background instead of cropping) — the fixed stride the Filmstrip
  // windowing needs.
  const imgH = h - 4;
  const isWeak = gradeKey(p.grade) === 'weak';
  const rule = gradeRule(p.grade);
  return (
    <button
      data-sel={isSel ? '1' : '0'}
      onClick={() => onSelect(p.id)}
      className={cn(
        'group flex shrink-0 cursor-pointer flex-col gap-px border-0 p-px',
        'rounded-sm outline outline-2 transition-[colors,outline-color,transform] duration-fast ease active:scale-[.99]',
        isSel ? 'bg-raised' : 'bg-transparent hover:bg-surface hover:outline-line-strong',
        isSelected ? 'outline-mark' : isSel ? 'outline-ink' : 'outline-transparent',
      )}
    >
      <span className="relative block shrink-0 overflow-hidden bg-well" style={{ height: imgH, width: w }}>
        <Thumb path={p.path}
          className={cn('block h-full max-w-none transition-[filter] duration-fast ease',
            w ? 'w-full object-contain' : 'w-auto',
            'group-hover:brightness-110', isWeak && 'opacity-reject')}/>
        {isUsed && (
          <span className="absolute left-px top-px rounded-sm bg-well px-px">
            <Flag size={7} className="text-ink-2"/>
          </span>
        )}
        {isSelected && (
          <span className="absolute right-px top-px flex h-3 w-3 items-center justify-center rounded-sm bg-mark">
            <svg width="7" height="7" viewBox="0 0 24 24" fill="none" stroke={T.well} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20,6 9,17 4,12"/></svg>
          </span>
        )}
      </span>

      {/* Same rule as the contact sheet, so both views speak one language. */}
      <span aria-hidden className="block w-full" style={{ height: 'var(--rule)', background: rule ?? undefined }}/>

      {showFn && (
        <span className="flex w-full items-center gap-px">
          <span className={cn('t-num flex-1 truncate text-left text-xs',
                              isSel ? 'text-ink-2' : 'text-ink-3')}>
            {(p.path.split(/[\\/]/).pop() ?? '').replace(/\.[^.]+$/, '')}
          </span>
          {p.has_annotations && <AnnotatedMark/>}
          {p.stars > 0 && (
            <svg width="6" height="6" viewBox="0 0 24 24" fill={T.mark} stroke={T.mark} strokeWidth="2" className="shrink-0">
              <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
            </svg>
          )}
        </span>
      )}
    </button>
  );
});

/* ── Filmstrip ────────────────────────────────────────────────────
 * Horizontal scrubbing strip under the loupe. Windowed along the X axis:
 * it used to mount one FilmThumb per filtered photo (the whole library in
 * loupe mode — 21,416 nodes live), so now only the frames near the
 * viewport (±24) exist in the DOM, with spacer divs preserving the scroll
 * width. Cells are fixed 3:2 slots with object-contain — the same "never
 * crop the composition" rule as the contact sheet — because a fixed
 * stride is what makes index-accurate windowing and auto-scroll possible. */
export function Filmstrip({
  photos, selId, onSelect, usedPaths, selectedIds, h = 84, showFn = true,
}: {
  photos: any[]; selId: string | null; onSelect: (id: string) => void;
  usedPaths: Set<string>; selectedIds: Set<string>; h?: number; showFn?: boolean;
}) {
  const GAP = 4;
  const OVERSCAN = 24;
  const ref = useRef<HTMLDivElement | null>(null);
  const [scrollLeft, setScrollLeft] = useState(0);
  const [vw, setVw] = useState(0);
  const imgH = h - 4;
  const slotW = Math.round(imgH * 1.5);   // fixed 3:2 slot width
  const stride = slotW + 2 + GAP;         // + the button's 1px padding each side

  /* Measure the strip viewport (panel resizes and thumb-size tweaks included). */
  useEffect(() => {
    const el = ref.current; if (!el) return;
    const ro = new ResizeObserver(() => setVw(el.clientWidth));
    ro.observe(el); setVw(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const count = Math.ceil(vw / stride) + OVERSCAN * 2;
  const windowed = vw > 0 && photos.length > count;
  const start = windowed ? Math.max(0, Math.floor(scrollLeft / stride) - OVERSCAN) : 0;
  const end   = windowed ? Math.min(photos.length, start + count) : photos.length;

  /* Auto-scroll to the selected frame — by index, so it works even when the
   * selection sits outside the rendered window (the old querySelector-based
   * scroll silently did nothing in that case). */
  const selIdx = photos.findIndex(p => p.id === selId);
  useEffect(() => {
    const el = ref.current; if (!el || selIdx < 0) return;
    const target = selIdx * stride;
    if (target < el.scrollLeft || target + slotW > el.scrollLeft + el.clientWidth)
      el.scrollLeft = Math.max(0, target - el.clientWidth / 2 + slotW / 2);
  }, [selIdx, stride, slotW, vw]);

  return (
    <div ref={ref}
      onScroll={e => setScrollLeft(e.currentTarget.scrollLeft)}
      style={{ height: h + (showFn ? 18 : 0) + 12, overflowX:'auto', overflowY:'hidden',
               display:'flex', alignItems:'center', padding:'0 6px', gap: GAP }}>
      {start > 0 && <div aria-hidden style={{ width: start * stride, flexShrink: 0 }}/>}
      {photos.slice(start, end).map(p => (
        <FilmThumb key={p.id} p={p} w={slotW} isSel={p.id === selId} onSelect={onSelect}
          isUsed={usedPaths.has(p.path)} isSelected={selectedIds.has(p.id)} h={h} showFn={showFn}/>
      ))}
      {end < photos.length && <div aria-hidden style={{ width: (photos.length - end) * stride, flexShrink: 0 }}/>}
    </div>
  );
}

/* ── Grid View ──────────────────────────────────────────────────── */
export function GridView({
  photos, selId, onSelect, usedPaths, selectMode, setSelectMode, selectedIds, setSelectedIds, onCreateSequence, onAutoSequence,
  nicheDetecting = false, dupesCount, showDuplicates = false, onToggleDupes, shownCount = null,
}: {
  photos: any[]; selId: string | null; onSelect: (id: string) => void; usedPaths: Set<string>;
  selectMode: boolean; setSelectMode: (v: boolean) => void;
  selectedIds: Set<string>; setSelectedIds: React.Dispatch<React.SetStateAction<Set<string>>>;
  onCreateSequence: () => void; onAutoSequence: () => void;
  /** Ambient status pill props - presentation only, state stays in App. */
  nicheDetecting?: boolean; dupesCount?: number; showDuplicates?: boolean;
  onToggleDupes?: () => void; shownCount?: number | null;
}) {
  const toggleSelect = (id: string) => {
    setSelectedIds(prev => { const next = new Set(prev); next.has(id) ? next.delete(id) : next.add(id); return next; });
  };
  /* Row windowing: only the visible sheet rows (±3) exist in the DOM. rowExtra
   * = rule (2) + label strip (20) + row gap (12). */
  const [density, setDensity] = useState(220);
  const wg = useWindowedGrid({ itemCount: photos.length, minColWidth: density, gap: 12, rowExtra: 34 });
  return (
    <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', position:'relative' }}>
      {/* One quiet toolbar row: selection, density, dupes, ambient status and
          counts - everything the sheet needs, nothing it doesn't. */}
      <div className="flex h-8 shrink-0 items-center justify-between border-b border-line bg-surface px-2">
        <div className="flex items-center gap-2">
          <Button size="sm" variant={selectMode ? 'solid' : 'quiet'}
            onClick={() => { setSelectMode(!selectMode); setSelectedIds(new Set()); }}
            icon={<CheckSquare size={11}/>}
            >
            {selectMode ? `Select (${selectedIds.size})` : 'Select'}
          </Button>
          {selectMode && selectedIds.size > 0 && (
            <Button size="sm" variant="quiet" onClick={() => setSelectedIds(new Set())}
              >
              Clear
            </Button>
          )}

          <div className="h-4 w-px shrink-0 bg-line-strong"/>
          <label className="flex items-center gap-1">
            <span className="t-label">Size</span>
            <input type="range" min={140} max={420} step={20} value={density}
              onChange={e => setDensity(Number(e.target.value))}
              aria-label="Thumbnail size"
              className="range-token" style={{ width: 80 }}/>
          </label>

          {onToggleDupes && dupesCount !== undefined && dupesCount > 0 && (
            <>
              <div className="h-4 w-px shrink-0 bg-line-strong"/>
              <Button size="sm" variant={showDuplicates ? 'solid' : 'quiet'} onClick={onToggleDupes}
                title={showDuplicates ? 'Hide duplicate shots' : 'Show duplicate shots'}
                icon={<Copy size={10}/>}>
                Dupes <span className="t-num ml-1 opacity-70">{dupesCount}</span>
              </Button>
            </>
          )}

          {nicheDetecting && (
            <span role="status"
              className="glass animate-shimmer ml-1 flex items-center gap-1 rounded-md px-2 py-px text-xs text-ink-2">
              <span aria-hidden
                className="inline-block h-2 w-2 shrink-0 rounded-full border-2 border-current border-t-transparent"
                style={{ animation: 'spin .8s linear infinite' }}/>
              Detecting ideal niche…
            </span>
          )}
        </div>
        <span className="t-num text-xs text-ink-3">
          {shownCount != null && (<><span>{shownCount}</span> shown · </>)}
          {photos.length} photos
        </span>
      </div>

      {/* Contact sheet.
       *
       * A true grid, not a flex-wrap of natural-width images: every column is
       * the same width and both sheet edges finish flush, so the sheet reads as
       * one machined surface instead of a ragged scatter. Columns are
       * `minmax(0-scaled, 1fr)` via auto-fill, so the sheet is symmetric at any
       * window width — no half-visible column hanging off the right edge.
       *
       * Each frame sits in a 3:2 cell with `object-contain`: the composition is
       * never cropped (the previous aspect-crop silently re-framed every
       * vertical and square — a correctness bug in a tool whose whole job is
       * judging composition), and verticals letterbox symmetrically on the well
       * background rather than breaking the column rhythm. The label strip is a
       * fixed height so filenames, stars and scores align into columns down the
       * sheet. */}
      <div ref={wg.ref} onScroll={wg.onScroll} className="flex-1 overflow-auto bg-ground px-4 py-3">
        <div className="grid mx-auto gap-3"
          style={{ gridTemplateColumns: `repeat(${wg.cols}, minmax(0, 1fr))`,
                   maxWidth: wg.cols * 440,
                   paddingTop: wg.padTop, paddingBottom: wg.padBottom }}>
          {photos.slice(wg.first, wg.last).map(p => {
            const isChecked = selectedIds.has(p.id);
            const isUsed    = usedPaths.has(p.path);
            const isCurrent = p.id === selId && !selectMode;
            const rule      = gradeRule(p.grade);
            const badge     = gradeBadge(p.grade);
            const isWeak    = gradeKey(p.grade) === 'weak';
            const isPending = gradeKey(p.grade) === 'pending';
            return (
              <button key={p.id} onClick={() => selectMode ? toggleSelect(p.id) : onSelect(p.id)}
                className={cn(
                  'group relative flex cursor-pointer flex-col border-0 bg-transparent p-0',
                  'rounded-cell outline outline-2 outline-offset-2',
                  // Motion policy: transform only on anything holding a photo.
                  // The lift runs on the spring — a 2026 entrance curve for the
                  // one interaction every cell shares.
                  'transition-transform duration-spring ease-spring',
                  // Hover lift: the frame rises under the cursor.
                  'hover:-translate-y-0.5 active:scale-[.98]',
                  // Selection ring uses --focus: neutral interactivity, never warm.
                  isChecked || isCurrent ? 'outline-[color:var(--focus)]' : 'outline-transparent',
                )}
                style={{ contentVisibility: 'auto', containIntrinsicSize: '200px 190px' }}>
                <span className="relative block overflow-hidden rounded-t-cell bg-well" style={{ aspectRatio: '3/2' }}>
                  <Thumb path={p.path}
                    className={cn(
                      'block h-full w-full object-contain transition-[opacity,filter] duration-fast ease',
                      // Hover lift: the frame under the cursor brightens a step —
                      // luminance-only, so it never biases the colour read.
                      'group-hover:brightness-110',
                      // Weak frames physically sink — the cheapest high-value
                      // scanning affordance in the whole design.
                      isWeak && 'opacity-reject',
                      selectMode && !isChecked && 'opacity-reject',
                    )}/>
                  {/* The machine's verdict as a glass chip — top-left. Strong
                      speaks in the machine voice; Weak keeps the alarm
                      register; Mid stays silent. Enters on the spring.
                      Who speaks is gradeBadge's call, not this file's — see the
                      note there. Deciding it inline is how the chip came to
                      contradict the rule under the very same cell. */}
                  {badge && (
                    <span aria-hidden
                      className="t-label pointer-events-none absolute left-1 top-1 animate-chip-in rounded-md px-1 py-px"
                      style={{ background: T.glass, color: badge }}>
                      {gradeLabel(p.grade)}
                    </span>
                  )}
                  {/* The score is NOT repeated over the frame. It already has
                      two renderings on this cell: the number in the caption
                      row, which aligns into a scannable column, and the hover
                      bar below, which is preattentive. A third copy stamped on
                      the photograph obscured the picture to say what the strip
                      underneath already said. */}
                  {/* Hover score bar — the score you can read without reading.
                      A cold fill grows along the bottom edge of the frame,
                      spring-revealed on hover. Length = score, so a row of
                      frames sorts itself visually under the cursor. */}
                  {!isPending && p.score > 0 && (
                    <span aria-hidden
                      className="pointer-events-none absolute inset-x-0 bottom-0 h-rule origin-left bg-ai opacity-0 transition-[opacity,transform] duration-spring ease-spring group-hover:opacity-100"
                      style={{ transform: `scaleX(${Math.max(0.04, Math.min(1, p.score))})` }}/>
                  )}
                  {selectMode && (
                    <span className={cn(
                      'absolute left-1 top-1 flex h-4 w-4 items-center justify-center rounded-sm border',
                      'transition-colors duration-fast ease',
                      isChecked ? 'border-mark bg-mark' : 'border-ink-2 bg-well',
                    )}>
                      {isChecked && <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke={T.well} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20,6 9,17 4,12"/></svg>}
                    </span>
                  )}
                  {isUsed && (
                    <span className="t-label absolute right-1 top-1 rounded-sm bg-well px-1 !text-ink-2"
                      >
                      Used
                    </span>
                  )}
                </span>

                {/* The machine's verdict: a rule in a fixed position, so runs of
                    Strong align into bands down the sheet and read without being
                    read. Mid shows nothing — silence is "no opinion". */}
                <span aria-hidden className={cn('block w-full', isPending && 'hatch-pending')}
                      style={{ height: 'var(--rule)', background: rule ?? undefined }}/>

                <span className={cn(
                  'flex items-center justify-between gap-1 rounded-b-sm px-1 transition-colors duration-fast ease',
                  isCurrent ? 'bg-raised' : 'bg-surface group-hover:bg-raised',
                )}
                style={{ height: 20, paddingTop: 1, paddingBottom: 1 }}>
                  <span className={cn('flex-1 truncate text-left text-xs',
                                      isWeak ? 'text-ink-4' : 'text-ink-2')}
                    >
                    {(p.path.split(/[\\/]/).pop() ?? '').replace(/\.[^.]+$/, '')}
                  </span>
                  {p.stars > 0 && (
                    <span className="shrink-0 leading-none" title={`${p.stars} of 5`}
                      >
                      <svg width="8" height="8" viewBox="0 0 24 24" fill={T.mark} stroke={T.mark} strokeWidth="2">
                        <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
                      </svg>
                    </span>
                  )}
                  {p.stars > 0 && <span className="t-num shrink-0 text-xs text-mark-ink">{p.stars}</span>}
                  {p.has_annotations && <AnnotatedMark/>}
                  {!isPending && p.score > 0 && (
                    <span className={cn('t-num shrink-0 text-xs', isWeak ? 'text-ink-4' : 'text-ink-3')}
                      >
                      {formatScore(p.score)}
                    </span>
                  )}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Selection actions. Floats over the sheet, so it is the one surface that
          earns real elevation rather than a border. */}
      {selectMode && selectedIds.size > 0 && (
        <div className="glass elev-3 animate-fade-in absolute bottom-4 left-1/2 z-50 flex -translate-x-1/2 items-center gap-3 whitespace-nowrap rounded-md border border-line-strong px-4 py-2">
          <span className="text-sm text-ink">
            <span className="t-num">{selectedIds.size}</span> selected
          </span>
          <div className="h-4 w-px bg-line-strong"/>
          <Button variant="solid" onClick={onCreateSequence} icon={<Layers size={11}/>}
            >
            Start sequence
          </Button>
          <Button onClick={onAutoSequence} icon={<RefreshCw size={11}/>}
            >
            Auto
          </Button>
        </div>
      )}
    </div>
  );
}
