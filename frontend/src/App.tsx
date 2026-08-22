import { useState, useEffect, useCallback, useMemo, useRef, memo } from "react";
import axios from "axios";
import {
  DndContext, closestCenter, KeyboardSensor, PointerSensor,
  useSensor, useSensors,
} from "@dnd-kit/core";
import {
  SortableContext, sortableKeyboardCoordinates,
  verticalListSortingStrategy, useSortable,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
  FolderOpen, Layers, FileDown, RefreshCw,
  ImageOff, X, Sparkles, Copy, Flag,
  LayoutGrid, RectangleHorizontal, SlidersHorizontal,
  Download, CheckSquare, ArrowUpDown, ArrowUp, ArrowDown,
  Wand2, Zap, Eye, EyeOff, Upload, Search,
} from "lucide-react";
import { Button } from "./components/ui/Button";
import { Chip } from "./components/ui/Chip";
import { Segmented } from "./components/ui/Segmented";
import { AnnotatedMark } from "./components/ui/GradeRule";
import { Modal } from "./components/ui/Modal";
import { Field, TextArea } from "./components/ui/Field";
import { StarRating } from "./components/ui/StarRating";
import { ExifPanel } from "./components/ExifPanel";
import { T, gradeRule, gradeKey, gradeLabel, formatScore } from "./theme/tokens";
import { cn } from "./lib/cn";

const isTauri = () => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

// Block any accidental external analytics / tracking calls — this is a fully offline app.
if (typeof window !== "undefined") {
  const _origFetch = window.fetch.bind(window);
  const _BLOCKED   = ["googleapis.com", "analytics", "sentry.io", "segment.io", "mixpanel", "hotjar"];
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const url = input.toString();
    if (_BLOCKED.some(h => url.includes(h))) return Promise.reject(new Error(`Blocked external request: ${url}`));
    return _origFetch(input, init);
  };
}

const API = import.meta.env.VITE_API_URL || (isTauri() ? "http://127.0.0.1:8000" : "http://127.0.0.1:8000");
const thumbUrl = (p: string) => `${API}/api/thumb?path=${encodeURIComponent(p)}`;
const photoUrl = (p: string) => `${API}/api/photo?path=${encodeURIComponent(p)}`;

/** Classify free system RAM into a readiness level for grading. `min` is the
 *  server's hard gate (below it a grade is refused); a +1.2 GB margin above that
 *  is treated as "tight" (grades, but may drop to lighter CLIP scoring). */
function ramReadiness(gs: any): {
  level: 'clear' | 'tight' | 'critical' | 'unknown';
  free: number | null; total: number | null; percent: number | null; readout: string; tip: string;
} {
  const free    = gs?.ram_free_gb ?? null;
  const total   = gs?.ram_total_gb ?? null;
  const percent = gs?.ram_percent ?? null;
  const min     = gs?.ram_min_gb ?? 1.8;
  if (free == null) return { level: 'unknown', free, total, percent, readout: '', tip: 'System memory unknown' };
  // Compact readout that mirrors Task Manager: % in use + GB available.
  const readout = percent != null
    ? `${percent.toFixed(0)}% · ${free.toFixed(1)} GB free`
    : `${free.toFixed(1)}${total != null ? ` / ${total.toFixed(1)}` : ''} GB`;
  const usedTip = percent != null ? ` (${percent.toFixed(0)}% in use — matches Task Manager)` : '';
  // "clear" requires at least 5 GB free: the SigLIP encode subprocess needs
  // ~2 GB RAM during model load plus the grade worker's baseline ~1 GB, leaving
  // 2 GB breathing room on a 5 GB machine. Below 5 GB is genuinely risky.
  const clearThresh = Math.max(min + 1.2, 5.0);
  if (free < min)           return { level: 'critical', free, total, percent, readout, tip: `Only ${free.toFixed(1)} GB free${usedTip} — grading needs ~${min} GB. Close some apps before grading.` };
  if (free < clearThresh)   return { level: 'tight',    free, total, percent, readout, tip: `${free.toFixed(1)} GB free${usedTip} — enough to grade, but close Chrome or other heavy apps first for a stable cull.` };
  return { level: 'clear', free, total, percent, readout, tip: `${free.toFixed(1)} GB free${usedTip} — clear to grade.` };
}

/** A setInterval that runs `fn` immediately, skips ticks while the tab is hidden,
 *  and refreshes once when the tab becomes visible again. Keeps background polling
 *  from running (and flooding the server log) when the app isn't on screen. */
function useGuardedInterval(fn: () => void, ms: number, deps: any[]) {
  useEffect(() => {
    fn();
    const id = setInterval(() => { if (!document.hidden) fn(); }, ms);
    const onVis = () => { if (!document.hidden) fn(); };
    document.addEventListener("visibilitychange", onVis);
    return () => { clearInterval(id); document.removeEventListener("visibilitychange", onVis); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

/** Strip traversal sequences and normalise separators before sending paths to the API. */
const sanitizePath = (raw: string): string =>
  raw.trim()
    .replace(/[\/\\]+/g, "/")   // normalise separators
    .split("/")
    .filter(seg => seg !== "..")  // drop traversal segments
    .join("/")
    .replace(/^\//, match => match); // preserve leading slash (absolute paths)

/* The second palette that used to live here is gone.
 *
 * `C` declared its own twenty literals — a blue-black ground (#0a0a0d), a
 * blue-grey ink ramp and a green/amber/red grade set — while src/theme/tokens.css
 * declared a neutral ground (#1a1a1a), a neutral ink ramp and a teal/silent/dim
 * grade set. Both were live at once, in the same window, so the app rendered two
 * different greys depending on which half of the file drew the pixel, and editing
 * tokens.css moved only half the UI.
 *
 * Everything now reads T.* from theme/tokens.ts, which returns var(--x) rather
 * than a value, so tokens.css is the only file where a colour is decided. */

/* The grade vocabulary. Everything downstream inherits from these, which is why
 * they are the first thing converted — see src/theme/tokens.ts.
 *
 * Mid is deliberately silent. It is the majority bucket, so an amber badge on
 * every Mid frame put colour on most of the grid and made the machine's least
 * confident verdict its loudest. Neutral ink and no rule instead. */
function gc(g: string) {
  const k = gradeKey(g);
  if (k === 'strong') return T.gradeStrong;
  if (k === 'weak')   return T.gradeWeak;
  if (k === 'mid')    return T.ink2;   // silent — neutral, never amber
  return T.ink3;
}

/* ── Vision-critique region guide ───────────────────────────────────────────
 * One teacher vocabulary shared by every overlay (heatmap glow, box callouts,
 * Analysis panel). Maps each Qwen spatial label → a quality tier, a plain-English
 * title, and a one-line coaching tip a photographer can act on.
 */
type RegionTier = 'strong' | 'refine' | 'fix';
const REGION_GUIDE: Record<string, { tier: RegionTier; title: string; tip: string }> = {
  anchor_subject:     { tier:'strong', title:'Strong anchor',      tip:'The eye lands here first — a clear subject grounds the frame.' },
  composition_anchor: { tier:'strong', title:'Composition anchor', tip:'This element structures the shot — placement is working.' },
  focal_point_miss:   { tier:'refine', title:'Focal point drifts',  tip:'Attention wanders here — simplify or re-frame to hold the eye.' },
  blown_highlight:    { tier:'fix',    title:'Highlights clipped',  tip:'Detail lost in the brights — lower exposure or recover in post.' },
  crushed_shadow:     { tier:'fix',    title:'Shadows crushed',     tip:'Detail lost in the darks — lift shadows to keep texture.' },
  motion_blur:        { tier:'fix',    title:'Motion blur',         tip:'Subject isn’t sharp — raise shutter speed to freeze motion.' },
};
const regionGuide = (label: string) =>
  REGION_GUIDE[label] ?? { tier: 'refine' as RegionTier, title: (label || 'region').replace(/_/g, ' '), tip: '' };
const tierColor = (t: RegionTier) =>
  t === 'strong' ? T.gradeStrong : t === 'fix' ? T.gradeWeak : T.ink2;
const tierIcon  = (t: RegionTier) => t === 'strong' ? '✓' : t === 'fix' ? '!' : '◐';
/* The heatmap is painted ON the photograph, so it may not reach for --mark:
 * that colour is reserved for marks the photographer made himself. This is the
 * one place the alarm tokens apply to image content rather than status chrome,
 * because `fix` and `refine` flag something he actually has to act on.
 *
 * These were #3fb950 / #f85149 / #d8a657 — GitHub's palette, three raw hex
 * literals sitting under a comment claiming they were "kept distinct from theme
 * tokens". Distinct from the tokens is exactly what a stray literal is. */
const tierHeat  = (t: RegionTier) =>
  t === 'strong' ? T.gradeStrong : t === 'fix' ? T.alarmCrit : T.alarmWarn;

/* gLow() mapped each grade to a 14% tinted badge background. Deleted: the
 * machine's verdict is a 2px rule under the frame, not a filled badge behind
 * text. gl() was a duplicate of gradeLabel() in theme/tokens.ts. */
// gIcon() lived here — it mapped grades to ✅ / ⚠️ / ❌. It had no callers left,
// and emoji-as-status is the clearest "generated interface" tell there is. The
// grade is carried by the rule under each frame instead. Do not reintroduce it.

const _SLOGANS: Array<[RegExp, string]> = [
  // Patterns match the backend's (model-agnostic) progress wording. Never put a
  // model name in the SLOGAN text — these strings are shown to the user.
  [/scanning folder|found \d+|already graded/i, "Pulling the contact sheet…"],
  [/checking image files|unusable images/i,     "Culling the camera-shake casualties…"],
  [/analyz|image analysis/i,                    "Reading the light in every frame…"],
  [/near-duplicate|marking duplicates/i,        "Picking the best frame from each burst…"],
  [/preparing deep analysis/i,                  "The photo editor is pulling up a chair…"],
  [/judging each photo|deep analysis ready/i,   "Studying composition, moment, and story…"],
  [/scoring image quality|quality scoring/i,    "Running the darkroom technical check…"],
  [/light and contrast/i,                       "Measuring the exposure…"],
  [/style reference|creative brief/i,           "Comparing against the reference portfolio…"],
  [/taste profile/i,                            "Recalling your editorial eye…"],
  [/refining composition/i,                     "Second shooter weighing in…"],
  [/sequenc|assigning grades|building your gallery/i, "Building the selects…"],
  [/combining scores/i,                         "Matching each frame to its genre…"],
  [/saving results|photo details/i,             "Filing the contact sheet…"],
];
function toSlogan(desc: string): string {
  if (!desc) return '';
  for (const [re, slogan] of _SLOGANS) {
    if (re.test(desc)) return slogan;
  }
  return desc;
}

function SortableItem({ id, children }: { id: string; children: React.ReactNode }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id });
  const style = transform
    ? { transform: CSS.Transform.toString(transform), transition, opacity: isDragging ? 0.5 : 1, zIndex: isDragging ? 10 : 1 }
    : {};
  return <div ref={setNodeRef} style={style} {...attributes} {...listeners}>{children}</div>;
}


const FilmThumb = memo(function FilmThumb({
  p, isSel, onSelect, isUsed, isSelected, h = 84, showFn = true,
}: { p: any; isSel: boolean; onSelect: (id: string) => void; isUsed: boolean; isSelected: boolean; h?: number; showFn?: boolean }) {
  // Frames keep their real proportions here too — a fixed row height with
  // natural width, so a vertical reads as a vertical while scrubbing.
  const imgH = h - 4;
  const isWeak = gradeKey(p.grade) === 'weak';
  const rule = gradeRule(p.grade);
  return (
    <button
      data-sel={isSel ? '1' : '0'}
      onClick={() => onSelect(p.id)}
      className={cn(
        'flex shrink-0 cursor-pointer flex-col gap-px border-0 p-px',
        'rounded-sm outline outline-2 transition-colors duration-fast ease',
        isSel ? 'bg-raised' : 'bg-transparent hover:bg-surface',
        isSelected ? 'outline-mark' : isSel ? 'outline-ink' : 'outline-transparent',
      )}
    >
      <span className="relative block shrink-0 overflow-hidden bg-well" style={{ height: imgH }}>
        <img src={thumbUrl(p.path)} alt="" decoding="async" loading="lazy"
          className={cn('block h-full w-auto max-w-none', isWeak && 'opacity-reject')}/>
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

/* StarRating moved to components/ui/StarRating.tsx. The copy that lived here
 * shadowed it, so the shared component sat imported by nobody while this one
 * painted stars in oklch(70% .18 72) — an amber about ten degrees in hue from
 * the old grade accent. At 11px over a photograph the two were the same colour,
 * carrying two unrelated meanings inside one cell. Stars are the photographer's
 * judgement, so they belong to --mark; the machine's grade gives up colour. */

/* ExifBlock moved to components/ExifPanel.tsx — see that file for why it is
 * grouped now. It was the last component still setting its values in
 * 'SF Mono', a macOS font that does not exist on this machine, so the one
 * panel that is entirely numbers was the one not set in the app's mono face. */

/* ── Export Modal ────────────────────────────────────────────────── */
function ExportModal({ photos, filterGrade, onClose }: { photos: any[]; filterGrade: string | null; onClose: () => void }) {
  const [xmpState, setXmpState] = useState<'idle'|'busy'|'done'|'error'>('idle');
  const [xmpCount, setXmpCount] = useState(0);

  const handleDownload = (p: any) => {
    const a = document.createElement('a');
    a.href = photoUrl(p.path); a.download = p.path.split(/[\\/]/).pop() || 'photo.jpg';
    a.click();
  };
  const handleDownloadAll = () => photos.forEach((p, i) => setTimeout(() => handleDownload(p), i * 200));

  const handleExportXmp = async () => {
    setXmpState('busy');
    try {
      const res = await fetch(`${API}/api/export/metadata`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ photos: photos.map(p => ({
          path: p.path, grade: p.grade, score: p.score,
          critique: p.critique, breakdown: p.breakdown, nima_score: p.nima_score,
        })) }),
      });
      const data = await res.json();
      setXmpCount(data.exported ?? 0);
      setXmpState('done');
    } catch {
      setXmpState('error');
    }
  };

  return (
    <Modal
      title="Export photos"
      subtitle={<><span className="t-num">{photos.length}</span> photo{photos.length !== 1 ? 's' : ''}{filterGrade ? ` · ${filterGrade} only` : ''}</>}
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="solid" onClick={handleDownloadAll} icon={<Download size={11}/>}>
            Download all (<span className="t-num">{photos.length}</span>)
          </Button>
        </>
      }
    >
      {photos.map(p => (
        <div key={p.id} className="flex items-center gap-3 border-b border-line py-2 last:border-0">
          <img src={thumbUrl(p.path)} alt="" loading="lazy"
            className="block h-8 w-auto max-w-none shrink-0 rounded-sm bg-well"/>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm text-ink">{p.path.split(/[\\/]/).pop()}</p>
            <p className="t-num mt-px truncate text-xs text-ink-3">
              {[p.exif?.camera, p.exif?.aperture, p.exif?.shutter, p.exif?.iso ? `ISO ${p.exif.iso}` : null].filter(Boolean).join(' · ')}
            </p>
          </div>
          <Button size="sm" variant="quiet" onClick={() => handleDownload(p)}
            title={`Download ${p.path.split(/[\\/]/).pop()}`} icon={<Download size={10}/>}/>
        </div>
      ))}

      {/* XMP sidecars. State is carried by the words, not by a colour change —
          "Written" is unambiguous without turning the control green. */}
      <div className="sticky bottom-0 -mx-4 mt-2 flex items-center justify-between gap-3 border-t border-line bg-raised px-4 py-3">
        <div className="min-w-0">
          <p className="text-sm text-ink-2">XMP sidecars</p>
          <p className="mt-px text-xs text-ink-3">
            {xmpState === 'idle'  && 'Write .xmp files beside each photo — Lightroom and Capture One read them'}
            {xmpState === 'busy'  && 'Writing sidecars…'}
            {xmpState === 'done'  && <><span className="t-num">{xmpCount}</span> sidecar{xmpCount !== 1 ? 's' : ''} written beside your photos</>}
            {xmpState === 'error' && 'Export failed. Check the server log for the cause.'}
          </p>
        </div>
        <Button onClick={handleExportXmp} disabled={xmpState === 'busy'}
          variant={xmpState === 'error' ? 'danger' : 'solid'}>
          {xmpState === 'busy' ? 'Writing…' : xmpState === 'done' ? 'Written' : 'Export XMP'}
        </Button>
      </div>
    </Modal>
  );
}

/* ── Grid View ──────────────────────────────────────────────────── */
function GridView({
  photos, selId, onSelect, usedPaths, selectMode, setSelectMode, selectedIds, setSelectedIds, onCreateSequence, onAutoSequence,
}: {
  photos: any[]; selId: string | null; onSelect: (id: string) => void; usedPaths: Set<string>;
  selectMode: boolean; setSelectMode: (v: boolean) => void;
  selectedIds: Set<string>; setSelectedIds: React.Dispatch<React.SetStateAction<Set<string>>>;
  onCreateSequence: () => void; onAutoSequence: () => void;
}) {
  const toggleSelect = (id: string) => {
    setSelectedIds(prev => { const next = new Set(prev); next.has(id) ? next.delete(id) : next.add(id); return next; });
  };
  return (
    <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', position:'relative' }}>
      <div className="flex h-8 shrink-0 items-center justify-between border-b border-line bg-surface px-3">
        <div className="flex items-center gap-2">
          <Button size="sm" variant={selectMode ? 'solid' : 'quiet'}
            onClick={() => { setSelectMode(!selectMode); setSelectedIds(new Set()); }}
            icon={<CheckSquare size={11}/>}>
            {selectMode ? `Select (${selectedIds.size})` : 'Select'}
          </Button>
          {selectMode && selectedIds.size > 0 && (
            <Button size="sm" variant="quiet" onClick={() => setSelectedIds(new Set())}>
              Clear
            </Button>
          )}
        </div>
        <span className="t-num text-xs text-ink-3">{photos.length} photos</span>
      </div>

      {/* Contact sheet.
       *
       * Rows are a fixed height with each frame taking its NATURAL width. The
       * previous grid forced `aspectRatio:'3/2'` with `objectFit:'cover'`, which
       * silently re-cropped every vertical, 4:3 and in-camera square in the
       * library — so the composition being judged was not the one that was shot.
       * In a tool whose entire job is judging composition that is a correctness
       * bug, not a style one.
       *
       * The right edge is ragged rather than flush: true justification needs each
       * image's dimensions to solve row widths, and the client doesn't have them
       * without a load pass. Preserving aspect is the part that matters. */}
      <div className="flex-1 overflow-auto bg-ground p-2">
        <div className="flex flex-wrap gap-1">
          {photos.map(p => {
            const isChecked = selectedIds.has(p.id);
            const isUsed    = usedPaths.has(p.path);
            const isCurrent = p.id === selId && !selectMode;
            const rule      = gradeRule(p.grade);
            const isWeak    = gradeKey(p.grade) === 'weak';
            const isPending = gradeKey(p.grade) === 'pending';
            return (
              <button key={p.id} onClick={() => selectMode ? toggleSelect(p.id) : onSelect(p.id)}
                className={cn(
                  'group relative flex cursor-pointer flex-col border-0 bg-transparent p-0',
                  'rounded-sm outline outline-2 outline-offset-1 transition-[outline-color] duration-fast ease',
                  isChecked ? 'outline-mark' : isCurrent ? 'outline-ink' : 'outline-transparent',
                )}
                style={{ contentVisibility: 'auto', containIntrinsicSize: '180px 150px' }}>
                <span className="relative block overflow-hidden bg-well" style={{ height: 132 }}>
                  <img src={thumbUrl(p.path)} alt="" decoding="async" loading="lazy"
                    className={cn(
                      'block h-full w-auto max-w-none transition-opacity duration-fast ease',
                      // Weak frames physically sink — the cheapest high-value
                      // scanning affordance in the whole design.
                      isWeak && 'opacity-reject',
                      selectMode && !isChecked && 'opacity-reject',
                    )}/>
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
                    <span className="t-label absolute right-1 top-1 rounded-sm bg-well px-1 !text-ink-2">
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
                  'flex items-center gap-1 px-1 py-px',
                  isCurrent ? 'bg-raised' : 'bg-surface',
                )}>
                  <span className={cn('t-num flex-1 truncate text-left text-xs',
                                      isWeak ? 'text-ink-4' : 'text-ink-3')}>
                    {(p.path.split(/[\\/]/).pop() ?? '').replace(/\.[^.]+$/, '')}
                  </span>
                  {p.stars > 0 && (
                    <span className="shrink-0 leading-none" title={`${p.stars} of 5`}>
                      <svg width="8" height="8" viewBox="0 0 24 24" fill={T.mark} stroke={T.mark} strokeWidth="2">
                        <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
                      </svg>
                    </span>
                  )}
                  {p.stars > 0 && <span className="t-num shrink-0 text-xs text-mark-ink">{p.stars}</span>}
                  {p.has_annotations && <AnnotatedMark/>}
                  {!isPending && p.score > 0 && (
                    <span className={cn('t-num shrink-0 text-xs', isWeak ? 'text-ink-4' : 'text-ink-2')}>
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
        <div className="animate-fade-in absolute bottom-4 left-1/2 z-50 flex -translate-x-1/2 items-center gap-3 whitespace-nowrap rounded-md border border-line-strong bg-surface px-4 py-2 shadow-lg">
          <span className="text-sm text-ink">
            <span className="t-num">{selectedIds.size}</span> selected
          </span>
          <div className="h-4 w-px bg-line-strong"/>
          <Button variant="solid" onClick={onCreateSequence} icon={<Layers size={11}/>}>
            Start sequence
          </Button>
          <Button onClick={onAutoSequence} icon={<RefreshCw size={11}/>}>
            Auto
          </Button>
        </div>
      )}
    </div>
  );
}

/* ── Critique trigger parser ────────────────────────────────────── */
// Parses <trigger type="blur|heatmap|grid">text</trigger> tags emitted by the
// jury LLM into hoverable inline spans that drive the image overlay state.
// Falls back to plain text if the LLM produces no tags.
function parseCritique(
  text: string,
  onEnter: (type: string) => void,
  onLeave: () => void,
): React.ReactNode[] {
  const RE = /<trigger\s+type="([^"]+)">([^<]*)<\/trigger>/g;
  const nodes: React.ReactNode[] = [];
  let last = 0;
  let key  = 0;
  let m: RegExpExecArray | null;
  const COLORS: Record<string, string> = {
    blur:    T.alarmWarn,
    heatmap: T.alarmCrit,
    grid:    T.ink2,
  };
  while ((m = RE.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const typ   = m[1];
    const label = m[2];
    const col   = COLORS[typ] ?? T.ink2;
    nodes.push(
      <span key={key++}
        onMouseEnter={() => onEnter(typ)}
        onMouseLeave={onLeave}
        style={{ cursor:'pointer', textDecorationLine:'underline', textDecorationStyle:'dotted',
                 textUnderlineOffset:'2px', color:col, fontWeight:600 }}
      >{label}</span>
    );
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes.length ? nodes : [text];
}

function buildReasoningFromBreakdown(score: number, grade: string, breakdown: Record<string,number>): string {
  // Intentional low-light / chiaroscuro / soft-focus detection.
  // Strong Narrative intent alongside lower Lighting CLIP score = deliberate mood.
  const narrativeScore = (breakdown['Narrative'] ?? 0) as number;
  const lightingScore  = (breakdown['Lighting']  ?? 1.0) as number;
  const isMoody = narrativeScore >= 0.38 && lightingScore < 0.55;

  const NOTES_LIGHTING: [number,string][] = isMoody
    ? [[0.78,"Light has direction and authority — shadow play is doing the work."],[0.62,"Light is readable; contrast holds."],[0.45,"Low-key rendering — shadow weight reads as intentional mood."],[0,"Deep shadow dominance. Chiaroscuro or available-light approach — darkness as intent."]]
    : [[0.78,"Light has direction and authority — shadow play is doing the work."],[0.62,"Light is readable; contrast holds."],[0.45,"Flat light. No drama, no depth — nothing to push the subject forward."],[0,"Light is fighting the image. Blown highlights or dead-flat exposure."]];

  const NOTES_TECHNICAL: [number,string][] = isMoody
    ? [[0.78,"Technical execution disappears into the image — as it should."],[0.62,"Technically clean. No distraction."],[0.45,"Soft rendering or organic grain — intentional aesthetic signature, not a failure."],[0,"Technical compromise is visible — in fine-art work, intentional grain and glow are valid."]]
    : [[0.78,"Technical execution disappears into the image — as it should."],[0.62,"Technically clean. No distraction."],[0.45,"Some softness or exposure drift. Manageable, not invisible."],[0,"Technical failure is visible — motion blur, clipping, or heavy noise."]];

  const NOTES: Record<string, [number,string][]> = {
    Composition:     [[0.78,"Frame is airtight — every element earns its place."],[0.62,"Geometry works; the eye moves without fighting the edges."],[0.45,"Framing is serviceable but the edges carry dead weight."],[0,"Frame is loose — crop it or reshoot it."]],
    Lighting:        NOTES_LIGHTING,
    Narrative:       [[0.78,"The moment is decisive — gesture or tension frozen at exactly the right frame."],[0.62,"A moment caught, not staged — feels authentic."],[0.45,"Something is happening but nothing is at stake."],[0,"No moment. The scene is static and the camera just witnessed it."]],
    'Human/Culture': [[0.78,"The human subject commands the frame — presence is undeniable."],[0.62,"Human element adds weight; the figure belongs here."],[0.45,"Figures are present but incidental — they don't anchor anything."],[0,"No human element. Architectural or environmental — works only if intentional."]],
    Technical:       NOTES_TECHNICAL,
  };

  const tier = grade?.includes('Strong') ? 'strong' : grade?.includes('Weak') ? 'weak' : 'mid';
  // Use the actual graded score for display — this is the final fused pipeline value.
  // avgScore is kept for internal note selection only (moody/chiaroscuro detection).
  const bdVals = Object.values(breakdown).filter(v => typeof v === 'number') as number[];
  const avgScore = bdVals.length > 0 ? bdVals.reduce((s, v) => s + v, 0) / bdVals.length : score;
  const pct  = Math.round(score * 100);
  // Only rank real photographic aspects (keys with copy in NOTES). Excludes private
  // metadata and technical-audit fields so the Best/Weakest footer never names jargon.
  const sorted = Object.entries(breakdown)
    .filter(([k,v]) => typeof v === 'number' && k in NOTES)
    .sort((a,b) => b[1]-a[1]);
  const topKey    = sorted[0]?.[0]       ?? 'Narrative';
  const bottomKey = sorted.at(-1)?.[0]   ?? 'Technical';

  const VERDICT_LIGHTING_WEAK = isMoody
    ? "Atmospheric depth through shadow — low-key is the visual language here."
    : "Light is the problem here, not the solution.";
  const VERDICT_LIGHTING_MID = isMoody
    ? "Shadow and atmosphere doing most of the work — mood over exposure."
    : "Light is present but not working hard enough.";

  const VERDICT: Record<string, Record<string,string>> = {
    strong: { Narrative:"Street photographer's instinct — right place, right frame, right moment.", Composition:"Geometric authority. The structure carries the image.", Lighting:"Light as subject. Everything else serves the atmosphere.", 'Human/Culture':"The figure is the photograph. Everything else is context.", Technical:"Technically confident — the craft is invisible." },
    mid:    { Narrative:"The moment is there but the frame doesn't fully commit to it.", Composition:"Decent bones. The structure works but doesn't surprise.", Lighting: VERDICT_LIGHTING_MID, 'Human/Culture':"The human element is in the frame but not in control of it.", Technical:"Technically adequate. Won't lose the shot but won't win it either." },
    weak:   { Narrative:"No decisive moment — the shutter fired but nothing was caught.", Composition:"The frame is not resolved. Too much, too little, or in the wrong place.", Lighting: VERDICT_LIGHTING_WEAK, 'Human/Culture':"The subject is lost. Distance, angle, or timing killed it.", Technical:"Technical compromise dominates. The image can't recover from it." },
  };

  const LBL: Record<string,string> = { Narrative:'Moment', 'Human/Culture':'Human' };
  const verdict = VERDICT[tier]?.[topKey] ?? '';
  const lines: string[] = [tier[0].toUpperCase()+tier.slice(1)];
  if (verdict) lines.push(verdict);
  lines.push('');
  for (const [k,v] of sorted) {
    const note = (NOTES[k] ?? []).find(([t]) => v >= t)?.[1] ?? '';
    if (note) lines.push(`${LBL[k]??k}: ${note}`);
  }
  lines.push(`\nBest: ${LBL[topKey]??topKey}   ·   Weakest: ${LBL[bottomKey]??bottomKey}`);
  return lines.join('\n');
}

/* ── Aspect → canonical dimension classifier ───────────────────────
 * The Judge's Eye Evidence Checklist has five fixed photographic rows:
 *   tech → Focus · light → Exposure · human → Subject · auth → Moment · comp → Geometry
 * The Qwen primary grader emits niche-specific axis names (20 niches, ~70 distinct
 * axes), so a hardcoded lookup on 'Lighting'/'Human/Culture'/'Narrative' leaves rows
 * unrated. This maps every known axis (display Title-Case) onto a dimension; a keyword
 * fallback covers anything unseen. Where a niche genuinely lacks a dimension (e.g.
 * Landscape has no human axis) the row stays empty and falls back to a context label. */
const ASPECT_DIM: Record<string,'tech'|'light'|'human'|'auth'|'comp'> = {
  // canonical 5 (SpecVLM fallback + legacy)
  Technical:'tech', Composition:'comp', Lighting:'light', Narrative:'auth', 'Human/Culture':'human',
  // tech
  Detail:'tech', Execution:'tech', 'Depth Of Field':'tech', 'Detail Retention':'tech',
  Cleanliness:'tech', 'News Sharpness':'tech', 'Sharpness & Detail':'tech',
  // comp (geometry / framing / spatial)
  Geometry:'comp', 'Compositional Urgency':'comp', 'City Texture':'comp', 'Landscape Comp':'comp',
  'Depth Scale':'comp', 'Negative Space':'comp', 'Graphic Simplicity':'comp',
  'Visual Abstraction':'comp', 'Pattern Texture':'comp', 'Graphic Impact':'comp',
  'Framing':'comp', 'Geometry & Balance':'comp', 'Framing Instinct':'comp', 'Layered Depth':'comp',
  // light (lighting / mood / colour / atmosphere)
  'Light Atmosphere':'light', 'Light Quality':'light', 'Light Mood':'light', 'Nocturnal Mood':'light',
  'Color Palette':'light', 'Light Painting':'light', 'Color Form':'light', 'Tonal Balance':'light',
  Atmosphere:'light', 'Weather Drama':'light', Mood:'light',
  'Natural Light':'light', 'Mood & Tone':'light', 'Tonal Purity':'light', 'Contrast Purity':'light',
  'Available Light':'light', 'Natural Light Quality':'light',
  // human (subject / figure / expression / emotion)
  Human:'human', Expression:'human', 'Model Expression':'human', 'Subject Behavior':'human',
  'Subject Detail':'human', Emotion:'human', 'Emotional Moment':'human', 'Styling Aesthetic':'human',
  'Sense Of Place':'human', 'Subject Isolation':'human', 'Human Impact':'human',
  'Character Presence':'human', 'Emotional Resonance':'human', 'Scale Element':'human',
  Presence:'human', 'Scale & Life':'human',
  // auth (moment / narrative / concept / authenticity)
  Moment:'auth', 'Narrative Impact':'auth', Authenticity:'auth', Context:'auth', 'News Impact':'auth',
  'Cultural Authenticity':'auth', 'Urban Energy':'auth', 'Motion Quality':'auth', 'Temporal Effect':'auth',
  'Artistic Vision':'auth', 'Visual Poetry':'auth', 'Environmental Context':'auth', 'Habitat Context':'auth',
  'Editorial Mood':'auth', 'Peak Action':'auth', 'Story Telling':'auth', 'Conceptual Strength':'auth',
  'Visual Innovation':'auth', 'Intent Clarity':'auth',
  'Decisive Moment':'auth', 'Cultural Depth':'auth', 'Journalistic Integrity':'auth',
  'Narrative Suggestion':'auth', 'Conceptual Weight':'auth', Reduction:'auth', Immediacy:'auth',
  'Environmental Truth':'auth',
};
function aspectDim(label: string): 'tech'|'light'|'human'|'auth'|'comp'|'' {
  const hit = ASPECT_DIM[label];
  if (hit) return hit;
  const s = label.toLowerCase();
  if (/sharp|noise|technical|execution|clean|grain|render/.test(s)) return 'tech';
  if (/light|tonal|expos|contrast|atmos|mood|colou?r|nocturn|weather|night|chiaroscuro/.test(s)) return 'light';
  if (/human|subject|express|emotion|character|model|portrait|gesture|presence|behavio|figure|face|styling/.test(s)) return 'human';
  if (/moment|narrative|story|authentic|action|temporal|concept|news|energy|vision|urgenc|context|peak|innovation|intent/.test(s)) return 'auth';
  if (/compos|geometr|framing|negative|graphic|pattern|landscape|depth|abstract|place|texture|scale|spatial/.test(s)) return 'comp';
  return '';
}

/* ── Factor Annotations overlay ────────────────────────────────── */
const REGION_BOX: Record<string, [number,number,number,number]> = {
  'top-third':    [0,      0,      1,    0.33],
  'center':       [0.2,    0.2,    0.6,  0.6],
  'bottom-third': [0,      0.67,   1,    0.33],
  'full':         [0,      0,      1,    1],
  'left-half':    [0,      0,      0.5,  1],
  'right-half':   [0.5,    0,      0.5,  1],
  'top-left':     [0,      0,      0.5,  0.5],
  'top-right':    [0.5,    0,      0.5,  0.5],
  'bottom-left':  [0,      0.5,    0.5,  0.5],
  'bottom-right': [0.5,    0.5,    0.5,  0.5],
};
/* Annotation marks are the MACHINE's, not the photographer's, so they stay cold
 * or neutral — --mark is reserved for his own rings and stars, and a factor
 * overlay wearing it would claim his authorship. */
const FACTOR_COLORS: Record<string, string> = {
  blur:    T.alarmWarn,
  heatmap: T.alarmCrit,
  grid:    T.ink2,
};

const _INK  = T.ink;
const _SH   = `0 0 8px ${T.well}, 0 1px 3px ${T.well}`;
const _MONO = "'Courier New', monospace";

function FactorAnnotations({ factors }: { factors: any[] }) {
  if (!factors || factors.length === 0) return null;
  return (
    <svg
      style={{ position:'absolute', top:0, left:0, width:'100%', height:'100%', pointerEvents:'none' }}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <filter id="fa-txt" x="-5%" y="-20%" width="110%" height="140%">
          <feDropShadow dx="0" dy="0" stdDeviation="2.5" floodColor={T.well} floodOpacity="1"/>
        </filter>
      </defs>
      {factors.map((f: any, i: number) => {
        const [bx, by, bw, bh] = REGION_BOX[f.region] ?? REGION_BOX['full'];
        const isStrength = (f.impact ?? 0) > 0;
        const isWeakness = (f.impact ?? 0) < 0;
        const color = isStrength ? T.gradeStrong : isWeakness ? T.alarmCrit : (FACTOR_COLORS[f.type] ?? T.ink2);

        // Region in %, clamped so edges are always inside the frame
        const rx = bx * 100, ry = by * 100, rw = bw * 100, rh = bh * 100;
        const rx2 = rx + rw, ry2 = ry + rh;
        // Center point
        const cx = rx + rw / 2, cy = ry + rh / 2;

        // Label: inside top-left of region, always on-screen
        const lx = Math.max(1, Math.min(rx + 1.5, 70));
        const ly = Math.max(4, ry + 5);

        const impactAbs = Math.round(Math.abs(f.impact ?? 0) * 100);
        const badge = isStrength ? `[+${impactAbs}]` : isWeakness ? `[-${impactAbs}]` : null;
        const glyph = isStrength ? '●' : isWeakness ? '✕' : '○';

        return (
          <g key={i}>
            {/* Region fill wash */}
            <rect x={`${rx}%`} y={`${ry}%`} width={`${rw}%`} height={`${rh}%`}
              fill={color} fillOpacity="0.10" stroke="none"/>

            {/* Region border */}
            <rect x={`${rx}%`} y={`${ry}%`} width={`${rw}%`} height={`${rh}%`}
              fill="none" stroke={color} strokeWidth="1.8" strokeOpacity="0.75"
              strokeDasharray="6 3"/>

            {/* Strength: bold dashed ellipse around region */}
            {isStrength && (
              <ellipse cx={`${cx}%`} cy={`${cy}%`}
                rx={`${rw * 0.44}%`} ry={`${rh * 0.42}%`}
                fill="none" stroke={T.gradeStrong} strokeWidth="2.2"
                strokeOpacity="0.85" strokeDasharray="7 4"/>
            )}

            {/* Weakness: bold X-strike */}
            {isWeakness && (<>
              <line x1={`${rx + rw*0.06}%`} y1={`${ry + rh*0.06}%`}
                    x2={`${rx + rw*0.94}%`} y2={`${ry + rh*0.94}%`}
                stroke={T.alarmCrit} strokeWidth="2.2" strokeOpacity="0.8"/>
              <line x1={`${rx + rw*0.94}%`} y1={`${ry + rh*0.06}%`}
                    x2={`${rx + rw*0.06}%`} y2={`${ry + rh*0.94}%`}
                stroke={T.alarmCrit} strokeWidth="2.2" strokeOpacity="0.8"/>
            </>)}

            {/* Center dot */}
            <circle cx={`${cx}%`} cy={`${cy}%`} r="0.6%"
              fill={color} fillOpacity="1"/>

            {/* Connector: center dot → label */}
            <line x1={`${cx}%`} y1={`${cy}%`} x2={`${lx}%`} y2={`${ly}%`}
              stroke={color} strokeWidth="1" strokeOpacity="0.5" strokeDasharray="3 3"/>

            {/* Label: glyph + name + badge, inside region */}
            <text x={`${lx}%`} y={`${ly}%`}
              fill={color} fontSize="12" fontWeight="700"
              fontFamily="'Courier New', monospace"
              filter="url(#fa-txt)">
              {glyph} {f.label}{badge ? ` ${badge}` : ''}
            </text>

            {/* Note on second line */}
            {f.note && (
              <text x={`${lx}%`} y={`${Math.min(ly + 4, ry2 - 2)}%`}
                fill={color} fontSize="10" fillOpacity="0.75"
                fontFamily="'Courier New', monospace"
                filter="url(#fa-txt)">
                {f.note}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/* ── Analysis HUD — pen-notation corner annotation on the image ──── */
function AnalysisHUD({ grade, score, breakdown }: { grade: string; score: number; breakdown: Record<string,number> }) {
  const ASPECT_KEYS = ['Technical','Composition','Lighting','Narrative','Human/Culture'];
  const aspects = ASPECT_KEYS.map(k => [k, (breakdown[k] ?? 0)] as [string,number]);
  const maxV = Math.max(...aspects.map(([,v]) => v));
  const minV = Math.min(...aspects.map(([,v]) => v));
  const pct  = Math.round(score * 100);
  const gradeColor = gc(grade);
  return (
    <div style={{ position:'absolute', bottom:62, left:20, pointerEvents:'none', zIndex:2 }}>
      {/* Score + grade written directly on the photo */}
      <div style={{ display:'flex', alignItems:'baseline', gap:6, marginBottom:10 }}>
        <span style={{ fontFamily:_MONO, fontSize:'var(--text-xl)', fontWeight:700, lineHeight:1,
          color:gradeColor, textShadow:_SH, letterSpacing:'-.01em' }}>{pct}</span>
        <span style={{ fontFamily:_MONO, fontSize:'var(--text-xs)', color:_INK, textShadow:_SH, opacity:.45 }}>/100</span>
        <span style={{ fontFamily:_MONO, fontSize:'var(--text-xs)', fontWeight:700, letterSpacing:'.1em',
          color:gradeColor, textShadow:_SH, marginLeft:4 }}>{gradeLabel(grade).toUpperCase()}</span>
      </div>
      {/* Aspect scores as pen-ruled tick lines */}
      <div style={{ display:'flex', flexDirection:'column', gap:5 }}>
        {aspects.map(([k, v]) => {
          const vpct = Math.round((v as number) * 100);
          const isTop = v === maxV;
          const isBot = v === minV && v !== maxV;
          const col = isTop ? T.gradeStrong : isBot ? T.gradeWeak : _INK;
          const filled = vpct * 0.76;
          return (
            <div key={k} style={{ display:'flex', alignItems:'center', gap:8 }}>
              <span style={{ fontFamily:_MONO, fontSize:'var(--text-xs)', color:col, textShadow:_SH,
                width:82, textAlign:'right', flexShrink:0, opacity:.85, letterSpacing:'.02em' }}>
                {k}
              </span>
              <svg width="80" height="9" style={{ flexShrink:0, overflow:'visible' }}>
                <line x1="0" y1="4.5" x2="76" y2="4.5" stroke={`${col}`} strokeWidth="0.75" opacity="0.25"/>
                <line x1="0" y1="4.5" x2={filled} y2="4.5" stroke={col} strokeWidth="1.5" strokeLinecap="round"/>
                <line x1={filled} y1="1" x2={filled} y2="8" stroke={col} strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              <span style={{ fontFamily:_MONO, fontSize:'var(--text-xs)', fontWeight:700, color:col,
                textShadow:_SH, flexShrink:0, width:22 }}>{vpct}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Niche registry (mirrors src/niche_registry.py) ─────────────── */
const NICHE_GROUPS = [
  { category: "Street & Documentary", niches: [
    { key: "classic_street",  label: "Classic Street" },
    { key: "documentary",     label: "Documentary" },
    { key: "photojournalism", label: "Photojournalism" },
    { key: "travel_cultural", label: "Travel & Cultural" },
  ]},
  { category: "Architecture & Space", niches: [
    { key: "architectural", label: "Architectural" },
    { key: "liminal",       label: "Liminal / Atmospheric" },
    { key: "urban_city",    label: "Urban & City" },
  ]},
  { category: "Light & Mood", niches: [
    { key: "night",         label: "Night Photography" },
    { key: "long_exposure", label: "Long Exposure" },
    { key: "fine_art",      label: "Fine Art" },
    { key: "minimalist",    label: "Minimalist" },
  ]},
  { category: "Subject-Focused", niches: [
    { key: "portrait",          label: "Portrait" },
    { key: "wildlife",          label: "Wildlife" },
    { key: "fashion_editorial", label: "Fashion & Editorial" },
    { key: "sports_action",     label: "Sports & Action" },
    { key: "wedding",           label: "Wedding" },
  ]},
  { category: "Creative & Specialized", niches: [
    { key: "landscape",    label: "Landscape" },
    { key: "abstract",     label: "Abstract" },
    { key: "macro",        label: "Macro / Close-up" },
    { key: "experimental", label: "Experimental" },
  ]},
];

/* ── App ────────────────────────────────────────────────────────── */
export default function App() {
  const [folder,     setFolder]     = useState("");
  const [preset,     setPreset]     = useState("classic_street");
  const [photos,     setPhotos]     = useState<any[]>([]);
  const [carousel,   setCarousel]   = useState<any[]>([]);
  const [saved,      setSaved]      = useState<{name: string; sequence: any[]}[]>([]);
  const [loading,      setLoading]      = useState(false);
  const [listLoading,  setListLoading]  = useState(false);
  const [gradeProgress, setGradeProgress] = useState(0);
  const [gradeDesc,     setGradeDesc]     = useState("");
  // Encoder quality tier chosen for this run ("Fast" | "Balanced" | "Pro").
  // Deliberately a plain quality word — never a model name.
  const [gradeQuality,  setGradeQuality]  = useState("");
  const [gradeStartMs,  setGradeStartMs]  = useState<number | null>(null);
  const [gradeEtaSecs,  setGradeEtaSecs]  = useState<number | null>(null);
  const [toast,      setToast]      = useState<{msg: string; type: "success"|"error"|"info"} | null>(null);
  const [selId,      setSelId]      = useState<string | null>(null);
  const [nicheRec,   setNicheRec]   = useState<any>(null);
  const [nicheDetecting, setNicheDetecting] = useState(false);
  const [infoTab,    setInfoTab]    = useState<"exif"|"breakdown"|"analysis">("breakdown");
  const [scanMode,   setScanMode]   = useState(false);
  const [deepGrade,  setDeepGrade]  = useState(false);   // OFF = fast SigLIP zero-shot; ON = Qwen VLM (slower, GPU)
  const [graderUsed, setGraderUsed] = useState<'fast'|'deep'|'scan'|null>(null);  // which grader actually ran (transparency badge)
  const [mainTab,    setMainTab]    = useState<"gallery"|"duplicates"|"creative">("gallery");
  const [seqMode,    setSeqMode]    = useState<'auto'|'director'|'story'|'competition'>('story');
  const [directorPrompt,  setDirectorPrompt]  = useState('');
  const [directorResult,  setDirectorResult]  = useState<any>(null);
  const [directorLoading, setDirectorLoading] = useState(false);
  const [directorPool,    setDirectorPool]    = useState<any[]>([]);
  const [mogcoTarget]     = useState(5);   // default story length; story building lives in the Story tab, not culling
  const [mogcoMinScore,   setMogcoMinScore]   = useState(0.45);
  const [uploadLoading,   setUploadLoading]   = useState(false);
  const [uploadDragOver,  setUploadDragOver]  = useState(false);
  const [loupeMode,  setLoupeMode]  = useState<"loupe"|"grid">("loupe");
  const [subjType,   setSubjType]   = useState<string | null>(null);
  const [locked,     setLocked]     = useState<Set<string>>(new Set());
  const [used,       setUsed]       = useState<Set<string>>(new Set());
  const [redacted,      setRedacted]      = useState<Set<string>>(new Set());
  const [showDuplicates,setShowDuplicates] = useState(false);
  const [folders,      setFolders]      = useState<string[]>([]);
  const [browserMode,  setBrowserMode]  = useState<'open'|'add'>('open');
  const [catalogBanner,setCatalogBanner]= useState(false);
  const saveTimerRef       = useRef<ReturnType<typeof setTimeout> | null>(null);
  const skipFolderLoadRef  = useRef(false);
  const [showBrowser,setShowBrowser]= useState(false);
  const [bPath,      setBPath]      = useState("C:\\Users");
  const [bFolders,   setBFolders]   = useState<string[]>([]);
  const [bImages,    setBImages]    = useState<string[]>([]);
  const [bSelFolders, setBSelFolders] = useState<Set<string>>(new Set());
  const [lastBClick, setLastBClick] = useState<number | null>(null);
  const [bLoading,   setBLoading]   = useState(false);
  const [copied,     setCopied]     = useState(false);
  const [rightW,     setRightW]     = useState(280);
  const [filmThumbH, setFilmThumbH] = useState(84);
  const [showFilename,setShowFilename] = useState(true);
  const [showTweaks, setShowTweaks] = useState(false);
  const [filterGrade,setFilterGrade] = useState<string | null>(null);
  const [filterStars,setFilterStars] = useState<number | null>(null);
  const [sortScore,  setSortScore]   = useState<'desc'|'asc'|null>(null);
  const [exportModal,setExportModal] = useState(false);
  const [selectedIds,setSelectedIds] = useState<Set<string>>(new Set());
  const [selectMode, setSelectMode]  = useState(false);
  const [showStarSort, setShowStarSort] = useState(false);
  const [seqMinStars, setSeqMinStars]   = useState(0);
  const [dragOver,    setDragOver]      = useState(false);
  const [backendReady,   setBackendReady]   = useState(false);
  const [backendError,   setBackendError]   = useState(false);
  const [graderStatus,   setGraderStatus]   = useState<{last_mode:string,draft_available:boolean,verify_available:boolean,last_error:string|null,qwen_warm:boolean,qwen_loading:boolean,qwen_download_pct:number|null,warmup_done:boolean,warmup_running:boolean,compute_device?:string,vram_free_gb?:number|null,vram_total_gb?:number|null,gpu_name?:string|null,ram_free_gb?:number|null,ram_total_gb?:number|null,ram_min_gb?:number}|null>(null);
  // Live system-memory snapshot, polled every 2 s (see /api/system/ram) so the RAM
  // readiness indicator tracks Task Manager in real time rather than refreshing
  // only on modal open.
  const [sysRam, setSysRam] = useState<{ram_free_gb:number|null,ram_total_gb:number|null,ram_percent:number|null,ram_min_gb:number}|null>(null);
  const [preGradeModal,  setPreGradeModal]  = useState<{photoCount:number}|null>(null);
  const preGradeDialogRef = useRef<HTMLDivElement>(null);
  // Modal accessibility: Escape closes, Tab is trapped inside the dialog, and
  // focus is restored to the trigger on close (WCAG 2.1.2 / 2.4.3).
  useEffect(() => {
    if (!preGradeModal) return;
    const prevFocus = document.activeElement as HTMLElement | null;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { setPreGradeModal(null); return; }
      if (e.key !== 'Tab') return;
      const root = preGradeDialogRef.current;
      if (!root) return;
      const nodes = root.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );
      if (!nodes.length) return;
      const first = nodes[0], last = nodes[nodes.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('keydown', onKey); prevFocus?.focus?.(); };
  }, [preGradeModal]);
  const [rescanAll,      setRescanAll]      = useState(true);
  const [heatmapB64,     setHeatmapB64]     = useState<string | null>(null);
  const [heatmapPath,    setHeatmapPath]    = useState<string | null>(null);
  const [showHeatmap,    setShowHeatmap]    = useState(false);
  const [critTrigger,    setCritTrigger]    = useState<string>('');
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [pegFile,        setPegFile]        = useState<File | null>(null);
  const [pegHash,        setPegHash]        = useState<string | null>(null);
  const [pegLoading,     setPegLoading]     = useState(false);
  // ── Semantic search state ─────────────────────────────────────────────────
  const [searchQuery,    setSearchQuery]    = useState("");
  const [searchResults,  setSearchResults]  = useState<Set<string> | null>(null); // Set of paths
  const [searchLoading,  setSearchLoading]  = useState(false);
  // ── Jury critique state ───────────────────────────────────────────────────
  const [juryLoading,    setJuryLoading]    = useState(false);
  const [juryCritique,   setJuryCritique]   = useState<string | null>(null);
  const [juryThink,      setJuryThink]      = useState<string | null>(null);
  const [juryCritPath,   setJuryCritPath]   = useState<string | null>(null);
  // ── Engine health state ───────────────────────────────────────────────────
  const [engineHealth,        setEngineHealth]        = useState<{ status: "checking"|"online"|"offline"; missing: string[] }>({ status: "checking", missing: [] });
  const [ollamaPs,            setOllamaPs]            = useState<{name:string; size_vram:number; size_total:number}[]>([]);
  const [bannerDismissed,     setBannerDismissed]     = useState(false);
  const [isDownloading,       setIsDownloading]       = useState(false);
  const [downloadProgress,    setDownloadProgress]    = useState(0);
  const [currentDownloadModel,setCurrentDownloadModel]= useState("");
  const [downloadError,       setDownloadError]       = useState<string | null>(null);
  const [updateRequired,      setUpdateRequired]      = useState(false);
  // ── Creative Direction state ──────────────────────────────────────────────
  const [creativeAnchor,   setCreativeAnchor]   = useState<string | null>(null);
  const [creativePrompt,   setCreativePrompt]   = useState("");
  const [creativeMode,     setCreativeMode]     = useState<"canny"|"depth">("canny");
  const [creativeCount,    setCreativeCount]    = useState(5);
  const [creativeLoading,  setCreativeLoading]  = useState(false);
  const [creativeProgress, setCreativeProgress] = useState(0);
  const [creativeStage,    setCreativeStage]    = useState("");
  const [creativeResults,     setCreativeResults]     = useState<any[]>([]);
  const [creativeOutDir,      setCreativeOutDir]      = useState("");
  // Non-empty when the sequence was NOT art-directed: a score sort wearing a
  // story’s clothes. Shown, never swallowed.
  const [creativeFallback,    setCreativeFallback]    = useState("");
  const [creativeShowOriginal,setCreativeShowOriginal]= useState(false);
  const [usedCount,           setUsedCount]           = useState(0);
  const [sequenceSaving,      setSequenceSaving]      = useState(false);
  // ── PDF RAG state ─────────────────────────────────────────────────────────
  const [ragPdfs,       setRagPdfs]       = useState<{name:string,pages:number,phrases:number}[]>([]);
  const [ragUploading,  setRagUploading]  = useState(false);
  // ── Auditor / XAI overlay state ───────────────────────────────────────────
  const [isAuditModeActive,     setIsAuditModeActive]     = useState(false);
  const [reasoningOverlayUrl,   setReasoningOverlayUrl]   = useState<string | null>(null);
  const [reasoningOverlayPath,  setReasoningOverlayPath]  = useState<string | null>(null);
  const [showEyeOverlay,        setShowEyeOverlay]        = useState(false);
  const [photoNatDims,          setPhotoNatDims]          = useState<{w:number;h:number}|null>(null);
  const [deepCritique,          setDeepCritique]          = useState<{narrative_arc:string;geometry_composition:string}|null>(null);
  const [deepCritiqueLoading,   setDeepCritiqueLoading]   = useState(false);

  const filmRef    = useRef<HTMLDivElement>(null);
  const dragCounter = useRef(0);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  const notify = useCallback((msg: string, type: "success"|"error"|"info" = "info") =>
    setToast({ msg, type }), []);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 3200);
    return () => clearTimeout(t);
  }, [toast]);

  /* live ETA countdown while grading is in progress */
  const gradeProgressRef = useRef(gradeProgress);
  gradeProgressRef.current = gradeProgress;
  useEffect(() => {
    if (!loading || gradeStartMs === null) {
      setGradeEtaSecs(null);
      return;
    }
    const id = setInterval(() => {
      const p = gradeProgressRef.current;
      if (p <= 0.02) return;
      const elapsed = (Date.now() - gradeStartMs) / 1000;
      const total   = elapsed / p;
      setGradeEtaSecs(Math.max(0, Math.round(total - elapsed)));
    }, 1000);
    return () => clearInterval(id);
  }, [loading, gradeStartMs]);

  /* poll backend until it responds — shows loading screen until ready */
  useEffect(() => {
    let cancelled = false;
    let timerId: ReturnType<typeof setTimeout>;
    let attempts = 0;
    const check = () => {
      attempts++;
      if (attempts > 100) { // 60 s timeout
        if (!cancelled) setBackendError(true);
        return;
      }
      fetch(`${API}/`)
        .then(r => { if (r.ok && !cancelled) setBackendReady(true); })
        .catch(() => {
          if (!cancelled) timerId = setTimeout(check, 600);
        });
    };
    check();
    return () => { cancelled = true; clearTimeout(timerId); };
  }, []);

  /* poll Ollama engine health every 10 s */
  const fetchEngineHealth = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/health/engine`);
      if (r.ok) {
        const d = await r.json();
        const status = d.status ?? "offline";
        if (status === "offline") setBannerDismissed(false);
        setEngineHealth({ status, missing: d.missing_models ?? [] });
      } else {
        setBannerDismissed(false);
        setEngineHealth({ status: "offline", missing: [] });
      }
    } catch {
      setBannerDismissed(false);
      setEngineHealth({ status: "offline", missing: [] });
    }
  }, []);

  useGuardedInterval(fetchEngineHealth, 10_000, [fetchEngineHealth]);

  const fetchOllamaPs = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/ollama/status`);
      if (r.ok) {
        const d = await r.json();
        setOllamaPs(d.models ?? []);
      } else {
        setOllamaPs([]);
      }
    } catch {
      setOllamaPs([]);
    }
  }, []);

  useGuardedInterval(fetchOllamaPs, 15_000, [fetchOllamaPs]);

  // Live RAM poll — cheap psutil-only endpoint, every 2 s, paused while hidden.
  const fetchSysRam = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/system/ram`);
      if (r.ok) setSysRam(await r.json());
    } catch { /* leave last reading */ }
  }, []);
  useGuardedInterval(fetchSysRam, 2_000, [fetchSysRam]);

  const fetchRagStatus = useCallback(async () => {
    try {
      const resp = await axios.get<{phrases: string[], pdfs: {name:string,pages:number,phrases:number}[]}>(`${API}/api/rag/concepts`);
      setRagPdfs(resp.data.pdfs ?? []);
    } catch { /* silent */ }
  }, []);

  useEffect(() => { fetchRagStatus(); }, [fetchRagStatus]);

  /* download missing Ollama models one-by-one with live progress */
  const handleDownloadMissing = useCallback(async () => {
    const ollamaModels = engineHealth.missing.filter(m => !m.endsWith('.gguf'));
    if (ollamaModels.length === 0) return;
    setIsDownloading(true);
    setDownloadError(null);
    setUpdateRequired(false);
    try {
      for (const model of ollamaModels) {
        setCurrentDownloadModel(model);
        setDownloadProgress(0);
        let modelError: string | null = null;
        try {
          const resp = await fetch(`${API}/api/models/pull`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: model }),
          });
          if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
          const reader  = resp.body!.getReader();
          const decoder = new TextDecoder('utf-8');
          let buffer = '';
          outer: while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (const line of lines) {
              if (!line.trim()) continue;
              try {
                const data = JSON.parse(line);
                if (data.error) {
                  modelError = data.error;
                  if (data.error.includes('newer version') || data.error.includes('412')) {
                    console.warn('[pull] Ollama out of date:', data.error);
                    setUpdateRequired(true);
                  } else {
                    console.error('[pull] Ollama error:', data.error);
                  }
                  reader.cancel();
                  break outer;          // exit the while loop, not just the for loop
                }
                if (data.total && data.completed) {
                  setDownloadProgress(Math.round((data.completed / data.total) * 100));
                }
                if (data.status === 'success') setDownloadProgress(100);
              } catch { /* skip malformed chunk */ }
            }
          }
        } catch (e: any) {
          modelError = e?.message ?? 'Could not reach the writing engine. Start Ollama, then try again.';
        }
        if (modelError) {
          console.error('[pull] Error downloading', model, ':', modelError);
          setDownloadError(`Failed to download ${model}: ${modelError}`);
          return;   // abort remaining models, go straight to finally
        }
      }
    } finally {
      setIsDownloading(false);
      setCurrentDownloadModel('');
      setDownloadProgress(0);
      await new Promise(r => setTimeout(r, 2000));
      fetchEngineHealth();
    }
  }, [engineHealth.missing, fetchEngineHealth]);

  /* fetch grader model status on startup and after each grading run */
  const isDoneForStatus = !loading && photos.length > 0 && photos.some((p:any) => p.grade !== 'Pending');
  useEffect(() => {
    if (!backendReady) return;
    fetch(`${API}/api/models/status`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setGraderStatus(d); })
      .catch(() => {});
  }, [backendReady, isDoneForStatus, preGradeModal]);

  /* Poll status every 3 s while the pre-grade modal is open — keeps the RAM /
     engine readiness live so the user sees an up-to-date "clear to grade" state. */
  useEffect(() => {
    if (!preGradeModal) return;
    const id = setInterval(() => {
      fetch(`${API}/api/models/status`)
        .then(r => r.ok ? r.json() : null)
        .then(d => { if (d) setGraderStatus(d); })
        .catch(() => {});
    }, 3000);
    return () => clearInterval(id);
  }, [preGradeModal]);

  /* fetch excluded-photo count from the server */
  useEffect(() => {
    fetch(`${API}/api/creative-direction/used-count`)
      .then(r => r.json())
      .then(d => setUsedCount(d.count ?? 0))
      .catch(() => {});
  }, []);

  const sel = useMemo(() => photos.find(p => p.id === selId) ?? photos[0] ?? null, [photos, selId]);

  useEffect(() => {
    if (photos.length > 0 && !selId) setSelId(photos[0].id);
  }, [photos]);

  /* Lazy EXIF fetch.
   *
   * "Has no EXIF yet" was the wrong test. A catalogue written before the reader
   * was rewritten holds the old sparse shape — often just date and time — and
   * that is non-empty, so the panel would show two rows forever and never ask
   * the server for the other twenty. `file_size` is the marker: it comes from
   * stat() rather than the file's EXIF block, so the current reader emits it for
   * every readable photo and no older record can have it. */
  useEffect(() => {
    if (!sel) return;
    const cached = sel.exif || {};
    if (Object.keys(cached).length > 0 && cached.file_size) return;
    const ctrl = new AbortController();
    axios.get(`${API}/api/exif`, { params: { path: sel.path }, signal: ctrl.signal })
      .then(r => {
        if (Object.keys(r.data).length > 0)
          setPhotos(prev => prev.map(p => p.id === sel.id ? { ...p, exif: r.data } : p));
      })
      .catch(e => { if (!axios.isCancel(e)) console.warn('[exif]', e); });
    return () => ctrl.abort();
  }, [sel?.id]);

  /* auto-scroll filmstrip to selected thumb */
  useEffect(() => {
    const el = filmRef.current; if (!el) return;
    const btn = el.querySelector('[data-sel="1"]') as HTMLElement | null; if (!btn) return;
    const er = el.getBoundingClientRect(), br = btn.getBoundingClientRect();
    if (br.left < er.left || br.right > er.right)
      el.scrollLeft += (br.left + br.width / 2) - (er.left + er.width / 2);
  }, [selId]);


  const filteredPhotos = useMemo(() => {
    const carouselPaths = new Set(carousel.map((c: any) => c.path));
    const base = photos.filter(p => {
      if (searchResults !== null && !searchResults.has(p.path)) return false; // semantic search filter
      if (!showDuplicates && redacted.has(p.path)) return false;   // non-best duplicates hidden unless toggled
      const starsOk = filterStars === null || p.stars === filterStars;
      if (filterGrade) return gradeLabel(p.grade) === filterGrade && starsOk;
      if (carouselPaths.has(p.path)) return true;                   // sequence photos always visible when no grade filter
      return starsOk;
    });
    if (!sortScore) return base;
    return [...base].sort((a, b) => sortScore === 'desc' ? b.score - a.score : a.score - b.score);
  }, [photos, filterGrade, filterStars, redacted, showDuplicates, sortScore, carousel, searchResults]);

  /* keyboard nav */
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (['INPUT','SELECT','TEXTAREA'].includes((document.activeElement as HTMLElement)?.tagName)) return;
      const ids = filteredPhotos.map(p => p.id);
      const i = ids.indexOf(selId ?? '');
      if (e.key === 'ArrowRight' || e.key === 'l') { e.preventDefault(); if (i < ids.length-1) setSelId(ids[i+1]); }
      if (e.key === 'ArrowLeft'  || e.key === 'h') { e.preventDefault(); if (i > 0) setSelId(ids[i-1]); }
      if (e.key >= '1' && e.key <= '5') {
        const n = parseInt(e.key);
        if (selId) handleSetStars(selId, sel?.stars === n ? 0 : n);
      }
      if (e.key === 'g' || e.key === 'G') setLoupeMode('grid');
      if ((e.key === 'e' || e.key === 'E') && isDone) setLoupeMode('loupe');
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [photos, selId, filteredPhotos]);

  /* clear creative state when folder changes */
  useEffect(() => {
    setCreativeResults([]);
    setCreativeAnchor(null);
    setCreativePrompt('');
    setCreativeOutDir('');
    setCreativeFallback('');
    setCreativeShowOriginal(false);
  }, [folder]);

  /* load photos when folder changes (skipped when resuming from catalog) */
  useEffect(() => {
    if (!folder.trim()) return;
    if (skipFolderLoadRef.current) { skipFolderLoadRef.current = false; return; }
    // Instant pre-grade niche recommendation — fire-and-forget in parallel with
    // the photo listing. Warm CPU CLIP on the server returns in <3 s; the picker
    // auto-selects the result and labels it "(Recommended)". Falls back silently
    // so a slow/failed detect never blocks folder loading.
    const safeFolder = sanitizePath(folder);
    setNicheRec(null);
    setNicheDetecting(true);
    axios.post(`${API}/api/recommend-niche`, { folder_path: safeFolder, folder_paths: [safeFolder] })
      .then(r => {
        if (r.data?.detected && r.data?.preset) { setNicheRec(r.data); setPreset(r.data.preset); }
      })
      .catch(() => {})
      .finally(() => setNicheDetecting(false));

    const load = async () => {
      setListLoading(true);
      try {
        const res = await axios.post(`${API}/api/list-folder`, { folder_path: sanitizePath(folder) });
        const rawPhotos: {path:string;exif:any}[] = res.data.photos || res.data.paths?.map((p: string) => ({path:p,exif:{}})) || [];
        if (!rawPhotos.length) notify("That folder has no images in it", "info");
        const ps = rawPhotos.map((p, i) => ({ id:`p-${i}`, path:p.path, grade:'Pending', score:0, breakdown:{}, critique:'', reasoning_log:'', is_verified:false, stars:0, exif:p.exif||{} }));
        setPhotos(ps);
        setFolders([folder]);
        setSelId(ps[0]?.id ?? null);
        setMainTab('gallery');
        setLoupeMode('grid');
      } catch (err: any) { notify(`${err.response?.data?.detail || "Could not read that folder"}`, "error"); }
      finally { setListLoading(false); }
    };
    load();
  }, [folder]);

  /* load flags */
  useEffect(() => {
    axios.get(`${API}/api/flags/load`)
      .then(r => { setLocked(new Set(r.data.locked||[])); setUsed(new Set(r.data.used||[])); })
      .catch(() => {});
  }, []);

  /* check for saved catalog on first load */
  useEffect(() => {
    axios.get(`${API}/api/catalog?t=` + Date.now())
      .then(r => { if (r.data.exists && r.data.photos?.length) setCatalogBanner(true); })
      .catch(() => {});
  }, []);

  /* auto-save catalog (debounced 2s) whenever graded photos or folder list changes */
  useEffect(() => {
    if (folders.length === 0 || !photos.some(p => p.grade !== 'Pending')) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      const photosToSave = photos.map(({ id: _id, ...rest }) => rest);
      axios.post(`${API}/api/catalog/save`, { photos: photosToSave, folders }).catch(() => {});
    }, 2000);
  }, [photos, folders]);

  // Hide heatmap overlay whenever the selected photo changes
  useEffect(() => { setShowHeatmap(false); }, [selId]);

  // Reset jury critique state when selected photo changes
  useEffect(() => {
    setJuryCritique(null);
    setJuryThink(null);
    setJuryCritPath(null);
    setCritTrigger('');
    setIsAuditModeActive(false);
    setShowEyeOverlay(false);
    setPhotoNatDims(null);
    setDeepCritique(null);
    setDeepCritiqueLoading(false);
  }, [selId]);

  // Disable overlay when leaving the analysis tab; user toggles it on manually
  useEffect(() => {
    if (infoTab !== 'analysis') setIsAuditModeActive(false);
  }, [infoTab]);

  // Poll for annotation readiness when a photo is selected but not yet annotated
  useEffect(() => {
    if (!sel || (sel.has_annotations && sel.eye_overlay_url !== undefined)) return;
    let cancelled = false;
    const check = async () => {
      if (cancelled || document.hidden) return;
      try {
        const stem = sel.path.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, '') ?? '';
        const r = await fetch(`${API}/api/annotations/${encodeURIComponent(stem)}`);
        if (!r.ok || cancelled) return;
        const d = await r.json();
        const gotAnnotations = d.has_annotations && d.score_factors?.length > 0;
        const gotOverlay     = Boolean(d.eye_overlay_url);
        if (gotAnnotations || gotOverlay) {
          setPhotos(prev => prev.map(p =>
            p.path === sel.path
              ? {
                  ...p,
                  ...(gotAnnotations ? {
                    has_annotations: d.has_annotations,
                    score_factors:   d.score_factors,
                  } : {}),
                  ...(gotOverlay ? { eye_overlay_url: d.eye_overlay_url } : {}),
                }
              : p
          ));
        }
      } catch { /* silent */ }
    };
    const id = setInterval(check, 8000);
    const onVis = () => { if (!document.hidden) check(); };
    document.addEventListener('visibilitychange', onVis);
    check();
    return () => { cancelled = true; clearInterval(id); document.removeEventListener('visibilitychange', onVis); };
  }, [selId, sel?.has_annotations]);

  const toggleHeatmap = useCallback(async () => {
    if (!sel) return;
    if (showHeatmap) { setShowHeatmap(false); return; }
    if (heatmapPath === sel.path && heatmapB64) { setShowHeatmap(true); return; }
    setHeatmapLoading(true);
    try {
      const resp = await axios.get<{b64: string}>(
        `${API}/api/heatmap/technical/${encodeURIComponent(sel.path)}`
      );
      setHeatmapB64(resp.data.b64);
      setHeatmapPath(sel.path);
      setShowHeatmap(true);
    } catch { /* silent fail */ } finally {
      setHeatmapLoading(false);
    }
  }, [sel, showHeatmap, heatmapPath, heatmapB64]);

  /* critique trigger hover — lazy-loads heatmap when blur/heatmap type is hovered */
  const handleTriggerEnter = useCallback(async (type: string) => {
    setCritTrigger(type);
    if ((type === 'blur' || type === 'heatmap') && sel && (heatmapPath !== sel.path || !heatmapB64)) {
      setHeatmapLoading(true);
      try {
        const resp = await axios.get<{b64: string}>(
          `${API}/api/heatmap/technical/${encodeURIComponent(sel.path)}`
        );
        setHeatmapB64(resp.data.b64);
        setHeatmapPath(sel.path);
      } catch { /* silent — overlay just won't show */ } finally {
        setHeatmapLoading(false);
      }
    }
  }, [sel, heatmapPath, heatmapB64]);

  const handleTriggerLeave = useCallback(() => setCritTrigger(''), []);

  const handleRagUpload = useCallback(async (file: File) => {
    setRagUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      await axios.post(`${API}/api/rag/upload`, form, { timeout: 300000 });
      await fetchRagStatus();
      notify(`"${file.name}" ingested as reference`, 'success');
    } catch {
      notify('PDF ingestion failed — check server logs', 'error');
    } finally {
      setRagUploading(false);
    }
  }, [fetchRagStatus]);

  const handleRagClear = useCallback(async () => {
    try {
      await axios.delete(`${API}/api/rag/clear`);
      setRagPdfs([]);
      notify('Reference library cleared', 'info');
    } catch {
      notify('Clear failed', 'error');
    }
  }, []);

  const handlePegUpload = useCallback(async (file: File) => {
    setPegFile(file);
    setPegHash(null);
    setPegLoading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const resp = await axios.post<{status: string, hash: string}>(`${API}/api/upload`, form);
      setPegHash(resp.data.hash);
    } catch {
      notify('Reference upload failed', 'error');
      setPegFile(null);
    } finally {
      setPegLoading(false);
    }
  }, []);

  const handleSemanticSearch = useCallback(async (q: string) => {
    if (!q.trim()) { setSearchResults(null); return; }
    setSearchLoading(true);
    try {
      const resp = await axios.get<{results: {hash: string; path: string; score: number}[]}>(
        `${API}/api/search/semantic`, { params: { q } }
      );
      const paths = new Set(resp.data.results.map(r => r.path));
      setSearchResults(paths);
      if (paths.size === 0) notify(`No results for "${q}"`, 'info');
    } catch {
      notify('Semantic search failed', 'error');
    } finally {
      setSearchLoading(false);
    }
  }, [notify]);

  const handleJuryCritique = useCallback(async (path: string) => {
    if (!path) return;
    // Use the path stem as image_hash (MD5 stem used by the ingestion pipeline)
    const hash = path.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, '') ?? '';
    if (!hash) return;
    if (juryCritPath === path && juryCritique) return; // already loaded
    setJuryLoading(true);
    setJuryCritique(null);
    setJuryThink(null);
    try {
      const resp = await axios.get<{critique: string; think?: string; error?: string}>(
        `${API}/api/critique/jury/${encodeURIComponent(hash)}`
      );
      if (resp.data.error) {
        // Backend returned a structured error (GGUF missing, model crash, etc.)
        setJuryCritique(`Critique failed: ${resp.data.error}`);
        console.error('[jury] backend error:', resp.data.error);
      } else {
        setJuryCritique(resp.data.critique);
        setJuryThink(resp.data.think ?? null);
        setJuryCritPath(path);
        if (resp.data.think) console.debug('[jury <think>]', resp.data.think);
      }
    } catch (err: any) {
      // Network-level failure (server offline, timeout, etc.)
      const detail = err?.response?.data?.error ?? err?.response?.data?.detail ?? null;
      setJuryCritique(detail ? `Critique failed: ${detail}` : 'Server unreachable — is the backend running?');
      console.error('[jury] network error:', err);
    } finally {
      setJuryLoading(false);
    }
  }, [juryCritPath, juryCritique]);

  /* folder browser */
  const loadBrowser = useCallback(async (path: string) => {
    setBLoading(true);
    try {
      const r = await axios.post(`${API}/api/browse-folder`, { folder_path: path });
      setBFolders(r.data.folders || []);
      setBImages(r.data.images || []);
    } catch { } finally { setBLoading(false); }
  }, []);

  const goUp = useCallback(() => {
    const parts = bPath.replace(/[\\/]+$/, '').split(/[\\/]/).filter(Boolean);
    if (parts.length <= 1) { setBPath('C:\\'); loadBrowser('C:\\'); return; }
    parts.pop();
    const p = parts.join('\\') || 'C:\\';
    setBPath(p); loadBrowser(p);
  }, [bPath, loadBrowser]);

  const handleBrowserFolderClick = useCallback((e: MouseEvent, path: string, _idx: number) => {
    const isCtrl = (e as any).ctrlKey || (e as any).metaKey;
    if (isCtrl) {
      // Ctrl+click toggles folder selection (for multi-add)
      setBSelFolders(prev => {
        const next = new Set(prev);
        if (next.has(path)) next.delete(path); else next.add(path);
        return next;
      });
    } else {
      // Single click navigates into the folder
      setBPath(path);
      loadBrowser(path);
      setBSelFolders(new Set());
    }
  }, [loadBrowser]);

  const openBrowser    = useCallback(() => { setBrowserMode('open'); setShowBrowser(true); loadBrowser(bPath); }, [bPath, loadBrowser]);
  const openAddFolder  = useCallback(() => { setBrowserMode('add');  setShowBrowser(true); loadBrowser(bPath); }, [bPath, loadBrowser]);

  // Populate the gallery from a /api/catalog payload (graded-photo checkpoint).
  // Shared by Resume and by grade-failure recovery. Returns the photo count.
  const applyCatalog = useCallback((data: any): number => {
    if (!data?.exists || !data?.photos?.length) return 0;
    const ps = data.photos.map((p: any, i: number) => ({ ...p, id: `p-${i}` }));
    // Apply same auto-redact logic as grading so duplicates are hidden in the gallery
    const autoRedacted = new Set<string>(
      ps.filter((p: any) => p.cluster_id >= 0 && !(p.sim_flag || '').startsWith('★'))
        .map((p: any) => p.path)
    );
    const firstVisible =
      ps.find((p: any) => !autoRedacted.has(p.path) && !((p.grade as string)?.includes('Weak'))) ??
      ps.find((p: any) => !autoRedacted.has(p.path));
    setPhotos(ps);
    setRedacted(autoRedacted);
    setSelId(firstVisible?.id ?? ps[0]?.id ?? null);
    return ps.length;
  }, []);

  const handleResume = useCallback(async () => {
    try {
      const r = await axios.get(`${API}/api/catalog?t=` + Date.now());
      if (!r.data.exists || !r.data.photos?.length) return;
      const savedFolders: string[] = r.data.folders || [];
      skipFolderLoadRef.current = true;
      setFolder(savedFolders[0] || '');
      setFolders(savedFolders);
      const n = applyCatalog(r.data);
      setLoupeMode('grid');
      setCatalogBanner(false);
      notify(`Resumed — ${n} photos from ${savedFolders.length} folder${savedFolders.length !== 1 ? 's' : ''}`, 'success');
    } catch { notify('Failed to resume session', 'error'); }
  }, [notify, applyCatalog]);

  const handleAddFolder = useCallback(async (newFolder: string) => {
    setListLoading(true);
    try {
      const res = await axios.post(`${API}/api/list-folder`, { folder_path: sanitizePath(newFolder) });
      const rawPhotos: {path:string;exif:any}[] = res.data.photos || [];
      setPhotos(prev => {
        const existing = new Set(prev.map(p => p.path));
        const added = rawPhotos
          .filter(p => !existing.has(p.path))
          .map((p, i) => ({ id:`p-${prev.length + i}`, path:p.path, grade:'Pending', score:0, breakdown:{}, critique:'', reasoning_log:'', is_verified:false, stars:0, exif:p.exif||{} }));
        return [...prev, ...added];
      });
      setFolders(prev => prev.includes(newFolder) ? prev : [...prev, newFolder]);
      notify(`Added ${rawPhotos.length} photos from ${newFolder.split(/[\\/]/).pop()}`, 'success');
    } catch { notify('❌ Failed to add folder', 'error'); }
    finally { setListLoading(false); }
  }, [notify]);

  const pickFolder = useCallback(async () => {
    const pw = (window as any).pywebview;
    if (pw?.api?.pick_folder) {
      const p: string|null = await pw.api.pick_folder();
      if (p) { setFolder(p); setPhotos([]); setSelId(null); }
      return;
    }
    if (isTauri()) {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const s = await open({ directory:true, multiple:false, title:"Select Photo Folder" });
      if (typeof s === 'string' && s) { setFolder(s); setPhotos([]); setSelId(null); }
      return;
    }
    try {
      const r = await axios.get(`${API}/api/pick-folder`);
      if (r.data?.path) { setFolder(r.data.path); setPhotos([]); setSelId(null); }
    } catch { notify("Could not open the folder picker. Paste a folder path instead.", "error"); }
  }, [notify]);

  /* grade — uses SSE stream so large folders never time out */
  const handleGrade = useCallback(async (forceRescan = false, skipModal = false) => {
    const safePath = sanitizePath(folder);
    if (!safePath && folders.length === 0) { notify("Enter a folder path, or use Open folder to browse.", "error"); return; }
    if (!skipModal) {
      const photoCount = photos.length > 0 ? photos.length : 0;
      setPreGradeModal({ photoCount });
      // Kick off Vision Engine preload as soon as modal opens so it's warm by the time
      // the user clicks Start Culling.
      if (!graderStatus?.qwen_warm && !graderStatus?.qwen_loading) {
        axios.post(`${API}/api/models/preload`).catch(() => {});
      }
      // NOTE: pre-grade niche auto-detection via a scan pass is disabled — the
      // scan runs the full scoring pipeline (seconds-per-image) under gpu_lock,
      // which could not meet the <3s target and blocked previews. If nicheRec is
      // already known (from a prior grade), the picker still pre-selects + labels
      // it; otherwise the user picks manually. See _scan_folder_for_data / the
      // /api/recommend-niche endpoint for the (currently unused) scan path.
      return;
    }
    setLoading(true);
    setGradeProgress(0);
    setGradeDesc("");
    setGradeStartMs(Date.now());
    setGradeEtaSecs(null);
    const allFolderPaths = folders.length > 0 ? folders.map(sanitizePath) : [safePath];
    try {
      const resp = await fetch(`${API}/api/grade/v2/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: allFolderPaths[0], folder_paths: allFolderPaths, preset, scan_mode: scanMode, deep_grade: deepGrade, force_rescan: forceRescan, mogco_target: mogcoTarget }),
      });
      if (!resp.ok) {
        try { const d = await resp.json(); throw new Error(d.error ?? `Server error ${resp.status}`); }
        catch (e: any) { if (e.message && !e.message.startsWith('{')) throw e; throw new Error(`Server error ${resp.status}`); }
      }
      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      const _readWithTimeout = (): Promise<ReadableStreamReadResult<Uint8Array>> =>
        Promise.race([
          reader.read(),
          new Promise<never>((_, reject) =>
            setTimeout(
              () => reject(new Error('No response from server for 45 s — the grader may have crashed. Check the server log and click Grade to retry.')),
              45_000,
            )
          ),
        ]);
      outer: while (true) {
        const { done, value } = await _readWithTimeout();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          let msg: any;
          try { msg = JSON.parse(line.slice(6)); } catch { continue; }
          if (msg.progress !== undefined) setGradeProgress(msg.progress);
          if (msg.desc) {
            // "Quality: Pro" is a one-off banner, not a stage — show it in the
            // badge rather than letting it flicker through the stage line.
            const q = /^Quality:\s*(.+)$/.exec(msg.desc);
            if (q) setGradeQuality(q[1].trim()); else setGradeDesc(msg.desc);
          }
          if (msg.quality) setGradeQuality(String(msg.quality));
          if (msg.error) throw new Error(msg.error);
          if (msg.done) {
            const ps = msg.data.map((p: any, i: number) => ({ ...p, id: `p-${i}` }));
            setPhotos(ps);
            setRedacted(new Set<string>(
              ps.filter((p: any) => p.cluster_id >= 0 && !(p.sim_flag || '').startsWith('★'))
                .map((p: any) => p.path)
            ));
            const firstVisible = ps.find((p: any) => !((p.grade as string)?.includes('Weak')))
              ?? ps[0];
            setSelId(firstVisible?.id ?? ps[0]?.id ?? null);
            // Populate carousel from MOGCO result if present, else clear it
            if (msg.mogco_sequence?.length > 0) {
              setCarousel(msg.mogco_sequence);
              setSubjType('mogco-beam');
            } else {
              setCarousel([]);
            }
            setMainTab('gallery');
            setLoupeMode('loupe');
            setInfoTab('breakdown');
            setLoading(false);
            setGradeProgress(0);
            setGradeDesc("");
            const mogcoNote = msg.mogco_sequence?.length > 0
              ? ` · ${msg.mogco_sequence.length}-slot sequence ready`
              : '';
            const mogcoErr  = msg.mogco_error
              ? ` · Sequence: ${msg.mogco_error}`
              : '';
            if (msg.mogco_error) notify(`${msg.mogco_error}`, 'error');
            // Transparency: report which grader actually ran, and warn (don't hide)
            // when Deep Grade was requested but silently fell back to Fast (SigLIP)
            // because there wasn't enough free RAM to load the vision model.
            const _graders = new Set<string>(ps.map((p: any) => p?.breakdown?._grader).filter(Boolean));
            const _usedDeep = _graders.has('qwen');
            setGraderUsed(scanMode ? 'scan' : _usedDeep ? 'deep' : 'fast');
            if (deepGrade && !scanMode && !_usedDeep) {
              notify('⚠️ Deep Grade fell back to Fast — not enough free memory for the deep analysis. Close some apps and re-grade for full accuracy.', 'error');
            }
            notify(`Graded ${msg.total} images${mogcoNote}${mogcoErr}`, 'success');
            axios.post(`${API}/api/recommend`, { photos: msg.data })
              .then(rec => setNicheRec(rec.data))
              .catch(() => {});
            break outer;
          }
        }
      }
    } catch (err: any) {
      // The grade stream failed — either the worker died (server emits an
      // explicit error) or the client read timed out (grader stalled). The
      // worker checkpoints completed grades to catalog.json before any crash,
      // so try to recover graded-so-far instead of dropping the user on a blank
      // gallery with a bare error.
      const msg = err?.message || 'Failed';
      const isStall = /No response from server/i.test(msg);
      try {
        const r = await axios.get(`${API}/api/catalog?t=` + Date.now());
        const n = applyCatalog(r.data);
        if (n > 0) {
          setMainTab('gallery');
          setLoupeMode('grid');
          notify(`Grading stopped early — recovered ${n} graded photo${n !== 1 ? 's' : ''} from the last checkpoint.`, 'error');
        } else {
          notify(isStall ? '❌ No response from the grader — it may be stalled. Check the server log and retry.' : `❌ ${msg}`, 'error');
        }
      } catch {
        notify(`${msg}`, 'error');
      }
    }
    setLoading(false);
    setGradeProgress(0);
  }, [folder, folders, preset, notify, applyCatalog]);

  /* generate sequence */
  const handleGenerate = useCallback(async () => {
    const pool = photos
      .filter(p => p.grade !== 'Pending')
      .filter(p => seqMinStars === 0 || (p.stars ?? 0) >= seqMinStars);
    const filterNote = seqMinStars > 0 ? ` rated ${seqMinStars}★+` : '';
    if (pool.length < 5) { notify(`A sequence needs at least 5 graded photos${filterNote}. Grade more, or clear the filter.`, 'error'); return; }
    setLoading(true);
    try {
      const res = await axios.post(`${API}/api/generate`, { photos: pool, seed: Math.floor(Math.random()*999999), avoid_paths: carousel.map((c: any) => c.path) });
      const d = res.data;
      setCarousel(Array.isArray(d) ? d : d.sequence);
      setSubjType(d.subject_type ?? null);
      setMainTab('gallery');
      notify('✅ Sequence generated', 'success');
    } catch (err: any) { notify(`${err.response?.data?.detail || "Could not build the sequence"}`, "error"); }
    setLoading(false);
  }, [photos, carousel, notify]);

  const handleExport = async () => {
    if (carousel.length < 5) return;
    try {
      const r = await axios.post(`${API}/api/editorial?fmt=portrait`, {
        photos: carousel.map(c => ({ path:c.path, grade:c.grade, score:c.score, breakdown:c.breakdown||{} })),
        excluded_paths: [],
      });
      const zip = r.data[0]?.zip;
      if (zip) { const a = document.createElement('a'); a.href = photoUrl(zip); a.download = 'Editorial_Carousel.zip'; a.click(); }
    } catch { notify('Export failed', 'error'); }
  };

  const handleSave = async () => {
    if (!carousel.length) return;
    const name = `Story ${saved.length + 1}`;
    try {
      await axios.post(`${API}/api/save-sequence`, { name, sequence: carousel });
      setSaved(prev => [...prev, { name, sequence: carousel }]);
      notify(`Saved as "${name}"`, 'success');
    } catch (err: any) { notify(`${err.response?.data?.detail || "Could not save the sequence"}`, 'error'); }
  };

  const handleDeleteSaved = useCallback((idx: number) => {
    setSaved(prev => prev.filter((_, i) => i !== idx));
  }, []);

  const handleRunCreativeDirection = useCallback(async () => {
    if (photos.length === 0) { notify('No photos loaded.', 'error'); return; }
    setCreativeLoading(true);
    setCreativeProgress(0);
    setCreativeStage('Initialising…');
    setCreativeResults([]);
    try {
      const resp = await fetch(`${API}/api/creative-direction/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          anchor_path:     creativeAnchor ? sanitizePath(creativeAnchor) : '',
          folder_path:     sanitizePath(folders[0] || folder),
          style_prompt:    creativePrompt,
          structure_mode:  creativeMode,
          n_target:        creativeCount,
          peg_image_hash:  pegHash ?? null,
          mode:            seqMode === 'auto' || seqMode === 'competition' ? seqMode : 'story',
        }),
      });
      if (!resp.ok) throw new Error(`Server error ${resp.status}`);
      const reader  = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      outer: while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          let msg: any;
          try { msg = JSON.parse(line.slice(6)); } catch { continue; }
          if (msg.progress !== undefined) setCreativeProgress(msg.progress);
          if (msg.desc)                   setCreativeStage(msg.desc);
          if (msg.error) throw new Error(msg.error);
          if (msg.done) {
            if (msg.data?.error) throw new Error(msg.data.error);
            const outputs = msg.data?.outputs ?? [];
            setCreativeResults(outputs);
            setCreativeOutDir(msg.data?.output_dir ?? '');
            setCreativeFallback(msg.data?.director_fallback ?? '');
            const ok = outputs.filter((r: any) => r.success).length;
            if (ok === 0 && outputs.length === 0) {
              notify('Creative Direction ran but produced no outputs.', 'info');
            } else if (msg.data?.director_fallback) {
              notify(`Picked the ${ok} highest-scoring photos — no art direction ran`, 'info');
            } else {
              notify(`Styled ${ok} of ${outputs.length} photos`, 'success');
            }
            break outer;
          }
        }
      }
    } catch (err: any) {
      notify(`Could not style the photos. ${err.message || err}`, 'error');
    } finally {
      setCreativeLoading(false);
      setCreativeProgress(0);
      setCreativeStage('');
    }
  }, [creativeAnchor, creativePrompt, creativeMode, creativeCount, photos, folder, folders, notify]);

  const handleSaveSequence = useCallback(async () => {
    const successes = creativeResults.filter((r: any) => r.success);
    if (!successes.length) return;
    setSequenceSaving(true);
    try {
      const resp = await fetch(`${API}/api/creative-direction/save-sequence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ outputs: successes, base_dir: creativeOutDir }),
      });
      const data = await resp.json();
      if (data.ok) {
        notify(`Saved ${data.count} photos to ${data.story_dir.split(/[\\/]/).pop()}`, 'success');
        setUsedCount(data.used_total ?? 0);
      } else {
        notify(`Could not save. ${data.error}`, 'error');
      }
    } catch (err: any) {
      notify(`Could not save. ${err.message}`, 'error');
    } finally {
      setSequenceSaving(false);
    }
  }, [creativeResults, creativeOutDir, notify]);

  const handleClearUsed = useCallback(async () => {
    try {
      const resp = await fetch(`${API}/api/creative-direction/clear-used`, { method: 'POST' });
      const data = await resp.json();
      if (data.ok) {
        setUsedCount(0);
        notify('History cleared — all photos eligible again', 'success');
      }
    } catch (err: any) {
      notify(`Could not clear. ${err.message}`, 'error');
    }
  }, [notify]);

  const handleSortByStars = useCallback((n: number) => {
    setCarousel(prev => [...prev].sort((a, b) => {
      const aS = a.stars ?? 0, bS = b.stars ?? 0;
      // Exact match to chosen star level floats to top, then descending
      const aMatch = aS === n ? 1 : 0, bMatch = bS === n ? 1 : 0;
      return bMatch !== aMatch ? bMatch - aMatch : bS - aS;
    }));
  }, []);

  const toggleFlag = useCallback(async (path: string, type: 'lock'|'used') => {
    try {
      await axios.post(`${API}/api/flags/${type}`, { path });
      const setter = type === 'lock' ? setLocked : setUsed;
      setter(prev => { const n = new Set(prev); n.has(path) ? n.delete(path) : n.add(path); return n; });
    } catch (err: any) { notify(`${err.response?.data?.detail || `Failed to toggle ${type}`}`, 'error'); }
  }, [notify]);

  const handleDragEnd = (e: any) => {
    if (e.active.id !== e.over?.id) {
      setCarousel(prev => {
        const a = [...prev];
        const oi = a.findIndex(i => i.path === e.active.id);
        const ni = a.findIndex(i => i.path === e.over.id);
        const [m] = a.splice(oi, 1); a.splice(ni, 0, m);
        return a;
      });
    }
  };

  const handleCopyPath = useCallback(() => {
    if (!sel?.path) return;
    navigator.clipboard.writeText(sel.path);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }, [sel]);

  const jumpToPhoto = useCallback((path: string) => {
    const p = photos.find(ph => ph.path === path);
    if (p) { setSelId(p.id); setMainTab('gallery'); }
  }, [photos]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    dragCounter.current = 0;
    setDragOver(false);
    const item = e.dataTransfer.items?.[0];
    const file = e.dataTransfer.files[0]; if (!file) return;
    const fullPath = (file as any).path as string | undefined;
    if (!fullPath) return;
    const entry = item?.webkitGetAsEntry?.();
    const isDir = entry?.isDirectory || fullPath.endsWith('/') || fullPath.endsWith('\\');
    const fp = isDir ? fullPath : fullPath.split(/[\\/]/).slice(0, -1).join('/') || fullPath;
    if (fp) { setFolder(fp); setPhotos([]); setSelId(null); }
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current++;
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current--;
    if (dragCounter.current === 0) setDragOver(false);
  }, []);

  const handleSetStars = useCallback((id: string, stars: number) => {
    setPhotos(prev => prev.map(p => p.id === id ? { ...p, stars } : p));
    // Fire-and-forget: train PersonalHead + queue DPO event
    const path = photos.find(p => p.id === id)?.path;
    if (path) {
      axios.post(`${API}/api/personal/star`, { path, stars }).catch(() => {});
    }
  }, [photos]);

  const handleCreateFromSelection = useCallback(() => {
    if (!selectedIds.size) { notify('Select photos first', 'error'); return; }
    const sel = photos.filter(p => selectedIds.has(p.id));
    setCarousel(sel);
    setSelectedIds(new Set());
    setSelectMode(false);
    setMainTab('sequence');
    notify('✅ Sequence created from selection', 'success');
  }, [photos, selectedIds, notify]);

  const onResizeDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const sx = e.clientX, sw = rightW;
    const onMove = (ev: MouseEvent) => setRightW(Math.max(200, Math.min(460, sw - (ev.clientX - sx))));
    const onUp = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
    window.addEventListener('mousemove', onMove); window.addEventListener('mouseup', onUp);
  }, [rightW]);

  const isGrading = loading;
  const isDone    = !loading && photos.length > 0 && photos.some(p => p.grade !== 'Pending');
  // If grading is reset/cleared, don't stay on a post-grade tab
  useEffect(() => {
    if (!isDone && mainTab !== 'gallery') setMainTab('gallery');
  }, [isDone, mainTab]);
  const picks     = photos.filter(p => gradeLabel(p.grade) === 'Strong' && !redacted.has(p.path)).length;
  const mids      = photos.filter(p => gradeLabel(p.grade) === 'Mid'    && !redacted.has(p.path)).length;
  // Paths marked as used: server flags + photos committed to any saved sequence
  const allUsedPaths = useMemo(() =>
    new Set([...Array.from(used), ...saved.flatMap(s => s.sequence.map((p: any) => p.path))]),
  [used, saved]);
  const rejects   = photos.filter(p => gradeLabel(p.grade) === 'Weak'    && !redacted.has(p.path)).length;
  // Star counts within the current grade filter (for the filter bar labels)
  const gradeFiltered = filterGrade ? photos.filter(p => gradeLabel(p.grade) === filterGrade) : photos;
  const starCounts = [0,1,2,3,4,5].map(n =>
    n === 0 ? gradeFiltered.filter(p => !p.stars).length
            : gradeFiltered.filter(p => p.stars === n).length
  );
  const selIdx    = filteredPhotos.findIndex(p => p.id === selId);
  const hasPrev   = selIdx > 0;
  const hasNext   = selIdx < filteredPhotos.length - 1;
  const isGraded  = isDone && sel && sel.grade !== 'Pending';

  const sequenceNarrative = useMemo(() => {
    if (!carousel.length) return null;
    const LMAP: Record<string,string> = {
      "Technical":"tech","News Sharpness":"tech","Cleanliness":"tech","Execution":"tech",
      "Detail Retention":"tech","Exposure":"tech","Sharpness & Detail":"tech",
      "Composition":"comp","Framing":"comp","Context":"comp","Geometry & Balance":"comp",
      "Negative Space":"comp","Framing Instinct":"comp","Layered Depth":"comp",
      "Lighting":"light","Atmosphere":"light","Natural Light":"light","Mood & Tone":"light",
      "Tonal Purity":"light","Contrast Purity":"light","Available Light":"light",
      "Natural Light Quality":"light",
      "Decisive Moment":"auth","Cultural Depth":"auth","Journalistic Integrity":"auth",
      "Narrative Suggestion":"auth","Conceptual Weight":"auth","Reduction":"auth",
      "Authenticity":"auth","Immediacy":"auth","Environmental Truth":"auth",
      "Subject Isolation":"human","Sense of Place":"human","Human Impact":"human",
      "Character Presence":"human","Emotional Resonance":"human","Scale Element":"human",
      "Human/Culture":"human","Presence":"human","Scale & Life":"human",
    };
    const tot: Record<string,number> = {tech:0,comp:0,light:0,auth:0,human:0};
    const cnt: Record<string,number> = {tech:0,comp:0,light:0,auth:0,human:0};
    carousel.forEach(c => {
      Object.entries(c.breakdown || {}).forEach(([lbl, val]) => {
        const k = LMAP[lbl];
        if (k && typeof val === 'number') { tot[k] += val; cnt[k]++; }
      });
    });
    const dimKeys = ['tech','comp','light','auth','human'] as const;
    const avg: Record<string,number> = {};
    dimKeys.forEach(k => { avg[k] = cnt[k] ? tot[k]/cnt[k] : 0; });
    const sorted = [...dimKeys].sort((a,b) => avg[b]-avg[a]);
    const strongest = sorted[0], weakest = sorted[sorted.length-1];
    const dimLabels: Record<string,string> = {
      tech:  'technical precision', comp: 'compositional instinct',
      light: 'atmospheric light',  auth: 'decisive-moment capture',
      human: 'human presence',
    };
    const nicheCtx: Record<string,string> = {
      'Classic Street':       'The sequence carries documentary hallmarks: authentic gesture, layered framing, and a sense of life caught mid-breath.',
      'Travel Editor':         'The sequence reads like a dispatched edit — cultural immersion, sense of place, and subjects genuinely encountered rather than posed.',
      'Photojournalism':       'The sequence holds documentary weight: technically grounded, contextually honest, anchored in authentic human stakes.',
      'Cinematic/Editorial':   'Light is the connective tissue. The sequence moves through moods rather than subjects — each frame builds atmosphere for the next.',
      'Fine Art/Contemporary': 'The sequence operates conceptually — compositional logic over candid impulse, tonal control over spontaneous capture.',
      'Minimalist/Urbex':      'Structure drives the sequence. Negative space and geometric restraint create rhythm without relying on human narrative.',
      'London Street':         'The sequence has the quality of a slow walk through a city at dusk — atmospheric, human, unhurried.',
      'Humanist/Everyday':     'The sequence is rooted in people. Dignity, proximity, and warmth thread through each frame.',
      'Landscape with Elements':'Light and environment carry the weight. The sequence breathes through its landscapes — foreground, depth, and tonal gradation.',
      'Snapshot / Point-and-Shoot': 'The sequence has the energy of unfiltered presence — raw, immediate, unconcerned with perfection.',
    };
    const niche = nicheRec?.preset ?? preset;
    const ctx   = nicheCtx[niche] ?? nicheRec?.reason ?? '';
    const parts: string[] = [];
    parts.push(`${carousel.length}-frame sequence evaluated against ${niche}.`);
    if (ctx) parts.push(ctx);
    if (avg[strongest] > 0) parts.push(`Dominant quality across the edit: ${dimLabels[strongest]}.`);
    if (weakest !== strongest && avg[weakest] > 0) parts.push(`Area with most room to grow: ${dimLabels[weakest]}.`);
    return parts.join(' ');
  }, [carousel, nicheRec, preset]);

  if (!backendReady) {
    return (
      <div className="fixed inset-0 flex flex-col items-center justify-center gap-4 bg-ground">
        {backendError ? (
          <>
            {/* An error explains what happened and what to do next. The previous
                version showed a ⚠️ over "Make sure the app is running correctly",
                which tells someone staring at a stopped app precisely nothing. */}
            <p className="t-label !text-alarm-crit">Not connected</p>
            <p className="max-w-[38ch] text-center text-sm text-ink">
              FrameGrade can't reach its engine, so nothing can be graded yet.
            </p>
            <p className="max-w-[42ch] text-center text-xs text-ink-3">
              It usually means the engine is still starting. Give it a few seconds and retry —
              if it keeps failing, close the app completely and open it again.
            </p>
            <Button className="mt-2" variant="solid"
              onClick={() => { setBackendError(false); window.location.reload(); }}>
              Retry
            </Button>
          </>
        ) : (
          <>
            <div style={{ width:40, height:40, border:`3px solid ${T.raisedHover}`, borderTopColor:T.ink3, borderRadius:'var(--r-round)', animation:'spin .8s linear infinite' }}/>
            <span style={{ fontSize:'var(--text-sm)', color:T.ink2, letterSpacing:'.05em' }}>Starting FrameGrade…</span>
          </>
        )}
      </div>
    );
  }

  return (
    <div
      style={{ display:'flex', flexDirection:'column', height:'100vh', background:T.ground, overflow:'hidden',
        fontFamily:"'Helvetica Neue',-apple-system,BlinkMacSystemFont,system-ui,sans-serif", fontSize:'var(--text-md)', color:T.ink }}
      onDrop={handleDrop} onDragOver={e => { e.preventDefault(); e.stopPropagation(); }} onDragEnter={handleDragEnter} onDragLeave={handleDragLeave}
    >

      {/* Drag-and-drop overlay */}
      {dragOver && (
        <div style={{ position:'fixed', inset:8, zIndex:200, pointerEvents:'none', borderRadius:'var(--r-md)',
          display:'flex', alignItems:'center', justifyContent:'center',
          background:T.scrim, backdropFilter:'blur(6px)',
          border:`2px dashed ${T.ink3}`,
        }}>
          <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:12 }}>
            <FolderOpen size={48} strokeWidth={1} style={{ color:T.ink2 }}/>
            <span className="text-md text-ink">Drop folder to load</span>
          </div>
        </div>
      )}

      {/* Toast */}
      {toast && (
        <div style={{
          position:'fixed', top:12, left:'50%', transform:'translateX(-50%)', zIndex:300,
          padding:'7px 16px', borderRadius:'var(--r-md)', fontSize:'var(--text-sm)', fontWeight:500, whiteSpace:'nowrap',
          // Success is silence: a completed action needs no green. Only an error
          // takes a hue, and only on the border — "everything is fine" is
          // expressed by the absence of colour, so colour here always means act.
          background: T.raised,
          border:`1px solid ${toast.type==='error' ? T.alarmCrit : T.lineStrong}`,
          color:T.ink, animation:'slideUp .3s cubic-bezier(.2,0,0,1)',
        }}>{toast.msg}</div>
      )}

      {/* Engine health warning banner */}
      {engineHealth.status !== "checking" && (() => {
        const missingOllama = engineHealth.missing.filter(m => !m.endsWith('.gguf'));
        const missingGguf   = engineHealth.missing.filter(m => m.endsWith('.gguf'));
        const isOffline     = engineHealth.status === "offline";

        // Model load state chips — gemma3:4b and qwen2.5vl:3b
        const VLM_TARGETS = ["gemma3:4b", "qwen2.5vl:3b"] as const;
        const MODEL_DISPLAY: Record<string, string> = {
          "gemma3:4b":    "Scene reader",
          "qwen2.5vl:3b": "Photo reader",
        };
        const modelChips = VLM_TARGETS.map(target => {
          const found = ollamaPs.find(m => m.name === target || m.name.startsWith(target.split(":")[0] + ":"));
          if (!found) return { label: target, display: MODEL_DISPLAY[target] ?? target, state: "absent" as const };
          const onGpu = found.size_vram > 0;
          return { label: target, display: MODEL_DISPLAY[target] ?? target, state: onGpu ? "gpu" as const : "cpu" as const, size_vram: found.size_vram };
        });
        const anyCpu    = modelChips.some(c => c.state === "cpu");
        const anyAbsent = modelChips.some(c => c.state === "absent");
        const showBanner = isOffline || missingOllama.length > 0 || missingGguf.length > 0 || anyCpu;
        if (!showBanner) return null;
        if (bannerDismissed && !isOffline) return null;

        return (
          <div style={{
            position:'fixed', top:0, left:0, right:0, zIndex:250,
            padding: isOffline ? '10px 18px' : '7px 16px',
            fontSize:'var(--text-sm)', fontWeight: isOffline ? 600 : 500,
            // Offline is the one banner the user must act on, so it takes the
            // alarm fill. A missing optional model is informational: neutral
            // surface, with the alarm colour reduced to a single hairline.
            background: isOffline ? T.alarmWarn : T.surface,
            borderBottom: isOffline ? `2px solid ${T.alarmCrit}` : `1px solid ${T.alarmWarn}`,
            color: isOffline ? T.well : T.ink2,
            display:'flex', alignItems:'center', gap:10, flexWrap:'wrap',
          }}>
            <span className="h-1 w-1 shrink-0 rounded-full" style={{ background: 'currentColor' }}/>
            {isOffline ? (
              <span style={{ flex:1, minWidth:0 }}>
                <strong>The writing engine isn't running.</strong>{' '}
                Grading still works. Written critiques, the creative director, and photo
                annotations are unavailable until you start Ollama.
              </span>
            ) : (
              <span style={{ flex:1, minWidth:0 }}>
                {missingOllama.length > 0 && <>Not installed yet: <strong>{missingOllama.map(m => MODEL_DISPLAY[m] ?? m).join(", ")}</strong>{missingGguf.length > 0 ? " · " : ""}</>}
                {missingGguf.length > 0 && <>Missing local file{missingGguf.length > 1 ? "s" : ""}: <strong>{missingGguf.join(", ")}</strong> — place manually in <code style={{ fontSize:'var(--text-xs)' }}>models/</code></>}
              </span>
            )}
            {/* Ollama out-of-date — overrides all other controls */}
            {updateRequired ? (
              <>
                <span style={{ flex:1, minWidth:0, fontSize:'var(--text-sm)', fontWeight:600, color:T.alarmCrit }}>
                  Your Ollama engine is out of date and cannot run these models.
                </span>
                <a
                  href="https://ollama.com/download"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ flexShrink:0, padding:'4px 12px', fontSize:'var(--text-sm)', fontWeight:700,
                    background:T.raised, color:T.ink,
                    border:`1px solid ${T.lineStrong}`, borderRadius:'var(--r-md)', cursor:'pointer',
                    whiteSpace:'nowrap', textDecoration:'none' }}>
                  Download Ollama Update
                </a>
              </>
            ) : (
              <>
                {/* Generic error */}
                {downloadError && !isDownloading && (
                  <span style={{ fontSize:'var(--text-xs)', color:T.alarmCrit, fontWeight:600, flex:1, minWidth:0 }}>
                    ✕ {downloadError}
                  </span>
                )}
                {/* Download / Retry button */}
                {missingOllama.length > 0 && !isDownloading && (
                  <button
                    onClick={() => { setDownloadError(null); handleDownloadMissing(); }}
                    style={{ flexShrink:0, padding:'4px 12px', fontSize:'var(--text-sm)', fontWeight:700,
                      background: T.raised,
                      color:T.ink, border:`1px solid ${downloadError ? T.alarmCrit : T.lineStrong}`,
                      borderRadius:'var(--r-md)', cursor:'pointer', whiteSpace:'nowrap' }}>
                    {downloadError ? 'Retry Download' : 'Download Missing Models'}
                  </button>
                )}
              </>
            )}
            {/* VLM model load state chips */}
            {!isOffline && (anyCpu || anyAbsent) && (
              <div style={{ display:'flex', alignItems:'center', gap:6, flexShrink:0, flexWrap:'wrap' }}>
                {modelChips.map(chip => {
                  const isGpu = chip.state === "gpu";
                  const isCpu = chip.state === "cpu";
                  return (
                    <span key={chip.label} title={isGpu ? `${chip.display} loaded in VRAM` : isCpu ? `${chip.display} running on CPU — VRAM headroom low` : `${chip.display} not loaded`}
                      style={{
                        padding:'2px 8px', borderRadius:'var(--r-sm)', fontSize:'var(--text-xs)', fontWeight:700, whiteSpace:'nowrap',
                        // A model sitting in VRAM is the expected case, so it is
                        // silent. Only CPU fallback — the case with a real cost
                        // the user can act on — reaches for the alarm token.
                        background: T.raised,
                        border: `1px solid ${isCpu ? T.alarmWarn : T.lineStrong}`,
                        color: isGpu ? T.ink2 : isCpu ? T.alarmWarn : T.ink3,
                      }}>
                      {chip.display} {isGpu ? '✓ GPU' : isCpu ? '⚡ CPU' : '—'}
                    </span>
                  );
                })}
                {anyCpu && <span style={{ fontSize:'var(--text-xs)', color:T.alarmWarn, fontWeight:500 }}>VRAM pressure — inference may be slow</span>}
              </div>
            )}
            {/* Progress indicator while downloading */}
            {isDownloading && (
              <div style={{ flexShrink:0, display:'flex', alignItems:'center', gap:8 }}>
                <div style={{ width:120, height:6, background:T.raised, borderRadius:'var(--r-sm)', overflow:'hidden' }}>
                  <div style={{ width:`${downloadProgress}%`, height:'100%', background:T.ink3, borderRadius:'var(--r-sm)', transition:'width .3s ease' }}/>
                </div>
                <span style={{ fontSize:'var(--text-xs)', whiteSpace:'nowrap', color:T.ink2 }}>
                  {currentDownloadModel}: {downloadProgress}% — do not close the app
                </span>
              </div>
            )}
            {/* Dismiss button */}
            {!isOffline && !isDownloading && (
              <button
                onClick={() => setBannerDismissed(true)}
                title="Dismiss"
                style={{ marginLeft:'auto', flexShrink:0, background:'none', border:'none', cursor:'pointer',
                  color: isOffline ? T.well : T.ink3, fontSize:'var(--text-md)', lineHeight:1, padding:'2px 4px' }}>
                ✕
              </button>
            )}
          </div>
        );
      })()}

      {/* Pre-grade info modal */}
      {preGradeModal && (
        <div style={{ position:'fixed', inset:0, zIndex:400, background:T.well, display:'flex', alignItems:'center', justifyContent:'center' }}
          role="presentation"
          onClick={() => setPreGradeModal(null)}>
          <div ref={preGradeDialogRef} style={{ background:T.surface1, border:`1px solid ${T.lineStrong}`, borderRadius:'var(--r-md)', padding:'28px 32px', maxWidth:420, width:'90%', display:'flex', flexDirection:'column', gap:16 }}
            role="dialog" aria-modal="true" aria-labelledby="pregrade-title"
            onClick={e => e.stopPropagation()}>
            <div id="pregrade-title" style={{ fontSize:'var(--text-md)', fontWeight:700, color:T.ink }}>Before you start</div>

            {/* Vision Engine status */}
            <div style={{ display:'flex', flexDirection:'column', gap:8 }}>
              {!graderStatus?.draft_available ? (
                <div className="rounded-sm border border-line-strong bg-raised px-3 py-2">
                  <div style={{ fontSize:'var(--text-sm)', fontWeight:700, color:T.ink, marginBottom:4 }}>
                    {graderStatus?.qwen_download_pct != null
                      ? `Downloading Vision Engine — ${graderStatus.qwen_download_pct}%`
                      : 'Vision Engine: downloading in background…'}
                  </div>
                  {graderStatus?.qwen_download_pct != null && (
                    <div style={{ height:4, background:T.raisedHover, borderRadius:'var(--r-sm)', overflow:'hidden', marginBottom:8 }}>
                      <div style={{ height:'100%', width:`${graderStatus.qwen_download_pct}%`,
                        background:T.ink3, borderRadius:'var(--r-sm)',
                        transition:'width .8s cubic-bezier(.2,0,0,1)' }}/>
                    </div>
                  )}
                  <div style={{ fontSize:'var(--text-sm)', color:T.ink2, lineHeight:1.5 }}>
                    ~6 GB one-time download — runs automatically in the background.
                    {graderStatus?.qwen_download_pct != null
                      ? ' Grading will start automatically once complete.'
                      : ' You can start grading now; it will begin once the download finishes.'}
                  </div>
                </div>
              ) : graderStatus?.qwen_warm ? (
                <div className="rounded-sm border border-line-strong bg-raised px-3 py-2">
                  <div style={{ fontSize:'var(--text-sm)', fontWeight:700, color:T.ink, marginBottom:4 }}>Vision Engine: warm and ready</div>
                  <div style={{ fontSize:'var(--text-sm)', color:T.ink2 }}>Already loaded in VRAM. Grading will start immediately.</div>
                </div>
              ) : graderStatus?.qwen_loading ? (
                <div className="rounded-sm border border-line-strong bg-raised px-3 py-2">
                  <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:6 }}>
                    <div style={{ width:10, height:10, borderRadius:'var(--r-round)', border:`2px solid ${T.ink3}`, borderTopColor:'transparent', animation:'spin .8s linear infinite', flexShrink:0 }}/>
                    <span className="text-sm text-ink">Vision Engine: loading into VRAM…</span>
                  </div>
                  <div style={{ fontSize:'var(--text-sm)', color:T.ink2 }}>
                    Loading model weights from disk. Takes <strong>~30–60 seconds</strong> — Start Culling will unlock automatically.
                  </div>
                </div>
              ) : (
                <div className="rounded-sm border border-line-strong bg-raised px-3 py-2">
                  <div className="mb-1 text-sm text-ink">Vision Engine: ready to load</div>
                  <div style={{ fontSize:'var(--text-sm)', color:T.ink2 }}>
                    Model is cached on disk. Loading starts now — will be ready in <strong>~30–60 seconds</strong>.
                  </div>
                </div>
              )}

              {/* System-RAM readiness — is it clear to grade? (live, polled every 2 s) */}
              {(sysRam || graderStatus) && (() => {
                const r = ramReadiness(sysRam ?? graderStatus);
                if (r.level === 'unknown') return null;
                // clear / tight / critical maps exactly onto silent / warn / crit.
                // "Clear to grade" needs no green tick: nothing being wrong is the
                // expected state, so it reads as neutral and only the two states
                // the user can act on carry a colour.
                const card = {
                  clear:    { accent:T.ink,       edge:T.lineStrong, title:`System memory: clear to grade`, body:`${r.free?.toFixed(1)} GB free — plenty of headroom for a full cull.` },
                  tight:    { accent:T.alarmWarn, edge:T.alarmWarn,  title:`System memory: tight but OK`,   body:`${r.free?.toFixed(1)} GB free. Grading will run, but may drop to lighter CLIP scoring. Closing a few apps gives the best results.` },
                  critical: { accent:T.alarmCrit, edge:T.alarmCrit,  title:`Low system memory`,             body:`Only ${r.free?.toFixed(1)} GB free — below the ~${(sysRam?.ram_min_gb ?? graderStatus?.ram_min_gb ?? 1.8)} GB needed. Close some apps before grading or the cull may be refused.` },
                }[r.level]!;
                return (
                  <div className="rounded-sm border bg-raised px-3 py-2" style={{ borderColor:card.edge }}>
                    <div style={{ fontSize:'var(--text-sm)', fontWeight:700, color:card.accent, marginBottom:4 }}>{card.title}</div>
                    <div style={{ fontSize:'var(--text-sm)', color:T.ink2, lineHeight:1.5 }}>{card.body}</div>
                  </div>
                );
              })()}

              {/* One-time INT4 quantisation disclaimer — only until the
                  pre-quantised cache exists on disk */}
              {graderStatus?.draft_available && !graderStatus?.qwen_int4_cached && !graderStatus?.qwen_warm && (
                <div className="rounded-sm border border-line-strong bg-raised px-3 py-2">
                  <div style={{ fontSize:'var(--text-sm)', fontWeight:700, color:T.ink, marginBottom:4 }}>
                    First cull: one-time engine optimisation
                  </div>
                  <div style={{ fontSize:'var(--text-sm)', color:T.ink2, lineHeight:1.5 }}>
                    The Vision Engine will be compressed for your GPU on this run — expect a
                    pause of <strong>a few minutes around 52%</strong>. The result is saved,
                    so every cull after this one skips it and starts in seconds.
                  </div>
                </div>
              )}

              {/* Pipeline calibration warmup status */}
              {graderStatus?.warmup_running && (
                <div className="rounded-sm border border-line-strong bg-raised px-3 py-2">
                  <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:4 }}>
                    <div style={{ width:9, height:9, borderRadius:'var(--r-round)', border:`2px solid ${T.ink3}`, borderTopColor:'transparent', animation:'spin .8s linear infinite', flexShrink:0 }}/>
                    <span style={{ fontSize:'var(--text-sm)', fontWeight:700, color:T.ink }}>Calibrating pipeline…</span>
                  </div>
                  <div style={{ fontSize:'var(--text-sm)', color:T.ink2 }}>Running your best photos through the engine to warm up CUDA kernels. Start Culling will unlock when done.</div>
                </div>
              )}
              {graderStatus?.warmup_done && !graderStatus?.warmup_running && (
                <div className="flex items-center gap-2 rounded-sm border border-line-strong bg-raised px-3 py-2">
                  <div style={{ width:7, height:7, borderRadius:'var(--r-round)', background:T.ink3, flexShrink:0 }}/>
                  <span style={{ fontSize:'var(--text-sm)', color:T.ink2 }}>Pipeline calibrated — first cull of this session will be fast</span>
                </div>
              )}

              {/* Re-grade toggle */}
              <Segmented
                value={rescanAll ? 'all' : 'new'}
                onChange={v => setRescanAll(v === 'all')}
                options={[
                  { value: 'all', label: 'Re-grade everything' },
                  { value: 'new', label: 'New photos only' },
                ]}
              />
              <p className="text-xs text-ink-3">
                {rescanAll
                  ? 'Every photo runs through the full pipeline.'
                  : 'Already-graded photos are skipped — only new additions are scored.'}
              </p>

              {/* Niche picker */}
              <div className="flex flex-col gap-1">
                <div className="t-label flex items-center gap-2">
                  Photography Niche
                  {nicheDetecting && (
                    <span className="flex items-center gap-1 normal-case tracking-normal text-ink-3">
                      <span style={{ width:9, height:9, borderRadius:'var(--r-round)', border:'2px solid currentColor', borderTopColor:'transparent', animation:'spin .8s linear infinite' }}/>
                      Detecting ideal niche…
                    </span>
                  )}
                  {!nicheDetecting && nicheRec?.detected && nicheRec?.preset === preset && (
                    <span className="normal-case tracking-normal text-ink-2">
                      ✓ auto-selected
                    </span>
                  )}
                </div>
                <select
                  value={preset}
                  aria-label="Grading niche / preset"
                  onChange={e => setPreset(e.target.value)}
                  className="w-full cursor-pointer rounded-sm border border-line-strong bg-raised px-2 py-1 text-sm text-ink outline-none focus-visible:border-focus">
                  {NICHE_GROUPS.map(g => (
                    <optgroup key={g.category} label={g.category}>
                      {g.niches.map(n => (
                        <option key={n.key} value={n.key}>
                          {n.label}{nicheRec?.preset === n.key ? '  (Recommended)' : ''}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              </div>

              {/* Photo count */}
              {preGradeModal.photoCount > 0 && (
                <p className="text-xs text-ink-3">
                  <span className="t-num">{preGradeModal.photoCount}</span> photo{preGradeModal.photoCount !== 1 ? 's' : ''} in folder
                </p>
              )}
            </div>

            {/* Deep Grade toggle — default OFF = fast SigLIP zero-shot; ON = Qwen VLM */}
            <label className="mt-1 flex cursor-pointer items-start gap-2 rounded-sm border border-line-strong bg-raised p-2">
              <input type="checkbox" checked={deepGrade} onChange={e => setDeepGrade(e.target.checked)}
                className="mt-px cursor-pointer" style={{ accentColor: T.ink }} />
              <div>
                <div className="text-sm text-ink">Deep grade</div>
                <div className="mt-px text-xs text-ink-3">
                  Off: fast grading — light on memory, recommended.
                  On: each photo is read in detail (more nuanced, slower, heavier on memory).
                </div>
              </div>
            </label>

            {/* Actions */}
            {(() => {
              const _notReady = graderStatus?.qwen_loading || graderStatus?.qwen_download_pct != null || graderStatus?.warmup_running;
              return (
                <div className="mt-1 flex justify-end gap-2">
                  <Button onClick={() => setPreGradeModal(null)}>
                    Cancel
                  </Button>
                  {/* The confirm was the accent-filled "primary" this design does
                      not have: emphasis is luminance, so it is simply the solid
                      variant sitting next to a bordered one. */}
                  <Button
                    variant="solid"
                    disabled={!!_notReady}
                    autoFocus
                    onClick={() => { setPreGradeModal(null); handleGrade(rescanAll, true); }}
                    icon={(graderStatus?.qwen_loading || graderStatus?.warmup_running)
                      ? <span style={{ width:10, height:10, borderRadius:'var(--r-round)', border:'2px solid currentColor', borderTopColor:'transparent', animation:'spin .8s linear infinite' }}/>
                      : undefined}>
                    {graderStatus?.qwen_loading ? 'Loading Engine…' : graderStatus?.warmup_running ? 'Calibrating…' : 'Start Culling'}
                  </Button>
                </div>
              );
            })()}
          </div>
        </div>
      )}

      {/* Export modal */}
      {exportModal && (
        <ExportModal
          photos={filterGrade ? filteredPhotos : photos}
          filterGrade={filterGrade}
          onClose={() => setExportModal(false)}
        />
      )}

      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="flex h-8 shrink-0 items-center gap-2 border-b border-line bg-surface px-3">

        <Button onClick={openBrowser} title="Open folder" icon={<FolderOpen size={13}/>}>
          {photos.length > 0 ? (folders.length > 1 ? `${folders.length} folders` : folder.split(/[\\/]/).pop()) : 'Open folder'}
        </Button>
        {photos.length > 0 && (
          <Button onClick={openAddFolder} title="Add another folder" variant="quiet"
            icon={<span className="text-md leading-none">+</span>}>
            Add folder
          </Button>
        )}

        <div className="flex-1"/>

        {/* Preset — hidden; value retained for grading logic */}

        {/* Model download. Progress is carried by a luminance fill behind the
            label rather than a coloured badge — it is informational, not an
            alarm, so it stays in the neutral register. */}
        {graderStatus?.qwen_download_pct != null && (
          <span className="relative inline-flex h-6 shrink-0 items-center gap-1 overflow-hidden rounded-sm border border-line-strong bg-surface px-2 text-ink-2">
            <span
              className="absolute inset-y-0 left-0 bg-raised-hover transition-[width] duration-slow ease"
              style={{ width: `${graderStatus.qwen_download_pct}%` }}
            />
            <span className="t-label relative !text-ink-3">Model</span>
            <span className="t-num relative text-xs">{graderStatus.qwen_download_pct}%</span>
          </span>
        )}

        {/* Grader mode indicator */}
        {graderStatus && (() => {
          const m = graderStatus.last_mode;
          const isIqaHeads = m === 'iqa_heads';
          const isClip     = m === 'clip_only';
          const isIdle     = m === 'idle' || !m;
          const dot   = isIqaHeads ? T.gradeStrong : isClip ? T.alarmWarn : T.ink3;
          const label = isIqaHeads ? 'Deep Edit' : isClip ? 'Scout Mode' : 'Ready';
          const tip   = graderStatus.last_error ? `Error: ${graderStatus.last_error}` :
                        isIqaHeads ? 'Full vision pipeline — composition, light, and moment scored' :
                        isClip     ? 'Fast contact-sheet pass — style matching only' :
                        'No grading run yet';
          if (isIdle) return null;
          void dot; // grader mode is informational, so it stays neutral
          return <Chip label={label} title={tip} />;
        })()}

        {/* GPU / CPU compute chip */}
        {graderStatus && (() => {
          const dev = graderStatus.compute_device;
          if (!dev) return null;
          const isGpu  = dev === 'gpu';
          const free   = graderStatus.vram_free_gb;
          const total  = graderStatus.vram_total_gb;
          const gpuName = graderStatus.gpu_name;
          const vramStr = free != null && total != null
            ? `${free.toFixed(1)} / ${total.toFixed(1)} GB VRAM`
            : free != null ? `${free.toFixed(1)} GB free` : '';
          const tip = isGpu
            ? [gpuName, vramStr].filter(Boolean).join(' · ')
            : 'Models running on CPU — no CUDA GPU detected or VRAM too low';
          // GPU is the expected state, so it is silent. Falling back to CPU is
          // the case worth flagging: it means a run will be far slower.
          return (
            <Chip
              label={isGpu ? 'GPU' : 'CPU'}
              title={tip}
              tone={isGpu ? 'neutral' : 'warn'}
              numeric={isGpu}
              value={isGpu && free != null ? `${free.toFixed(1)} GB` : undefined}
            />
          );
        })()}

        {/* System RAM chip — live (polled every 2 s), tells the user whether it's clear to grade */}
        {(sysRam || graderStatus) && (() => {
          const r = ramReadiness(sysRam ?? graderStatus);
          if (r.level === 'unknown') return null;
          // The one chip that has genuinely earned its colour on this machine:
          // two culls died tonight when free memory fell under the encoder's
          // load floor. Clear stays neutral so tight and critical actually read.
          const tone = ({ clear: 'neutral', tight: 'warn', critical: 'crit' } as const)[r.level];
          return <Chip label="RAM" title={r.tip} tone={tone} numeric value={r.readout} />;
        })()}

        {/* Grade filters. Mid carries a neutral dot rather than its own hue,
            matching the grid: Strong is marked, Weak is dimmed, Mid is silence. */}
        {isDone && (
          <Segmented
            className="animate-fade-in"
            value={filterGrade as 'Strong' | 'Mid' | 'Weak' | null}
            onChange={(v) => setFilterGrade(filterGrade === v ? null : v)}
            options={[
              { value: 'Strong', label: 'Strong', count: picks,   dot: T.gradeStrong },
              { value: 'Mid',    label: 'Mid',    count: mids,    dot: T.ink4 },
              { value: 'Weak',   label: 'Weak',   count: rejects, dot: T.gradeWeak },
            ]}
          />
        )}

        {isDone && graderUsed && (
          <Chip
            label={graderUsed === 'deep' ? 'Deep' : graderUsed === 'scan' ? 'Scan' : 'Fast'}
            title={graderUsed === 'deep'
              ? 'Deep grade: each photo was read in detail (highest accuracy).'
              : graderUsed === 'scan'
              ? 'Scan pass: quick look only, technical scoring skipped (fastest).'
              : 'Fast grade: standard quality scoring. Turn on Deep Grade (and free memory) for the detailed read.'}
          />
        )}

        {isDone && <div className="h-4 w-px shrink-0 bg-line-strong"/>}

        {isDone && (
          <Button
            variant={sortScore ? 'solid' : 'quiet'}
            onClick={() => setSortScore(s => s === null ? 'desc' : s === 'desc' ? 'asc' : null)}
            title={sortScore === 'desc' ? 'Sorted: Strong to Weak' : sortScore === 'asc' ? 'Sorted: Weak to Strong' : 'Sort by score'}
            icon={sortScore === 'desc' ? <ArrowDown size={11}/> : sortScore === 'asc' ? <ArrowUp size={11}/> : <ArrowUpDown size={11}/>}
          >
            Score
          </Button>
        )}

        {isDone && <div className="h-4 w-px shrink-0 bg-line-strong"/>}

        {/* Main views. Counts sit in the segment rather than inside the label
            string, so they render in tabular figures and the tab stops resizing
            as duplicates are resolved. */}
        {isDone && (() => {
          const dupCount = redacted.size > 0
            ? redacted.size
            : photos.filter(p => p.cluster_id >= 0 && !(p.sim_flag||'').includes('Best')).length;
          const madeCount = creativeResults.filter((r:any)=>r.success).length;
          return (
            <Segmented
              className="animate-fade-in"
              value={mainTab}
              onChange={(id) => { setMainTab(id); if (id === 'gallery') setLoupeMode('loupe'); }}
              options={[
                { value: 'gallery',    label: 'Gallery',    icon: <LayoutGrid size={11}/> },
                { value: 'duplicates', label: 'Duplicates', icon: <ImageOff size={11}/>,
                  count: dupCount > 0 ? dupCount : undefined },
                { value: 'creative',   label: 'Creative',   icon: <Wand2 size={11}/>,
                  count: madeCount > 0 ? madeCount : undefined },
              ]}
            />
          );
        })()}

        {isDone && mainTab === 'gallery' && (
          <Segmented
            iconOnly
            value={loupeMode}
            onChange={setLoupeMode}
            options={[
              { value: 'loupe', icon: <RectangleHorizontal size={12}/>, title: 'Loupe (E)' },
              { value: 'grid',  icon: <LayoutGrid size={12}/>,          title: 'Grid (G)' },
            ]}
          />
        )}

        {isDone && (
          <Button onClick={() => setExportModal(true)} icon={<Download size={11}/>}>
            Export
          </Button>
        )}

        {isDone && (
          <Button
            icon={<ArrowUpDown size={11}/>}
            title="Move files on disk into Strong / Mid / Weak folders"
            onClick={async () => {
              try {
                const res = await axios.post(`${API}/api/manage/sort-files`, {
                  folder_path: folders[0] || folder,
                  gallery: photos,
                  copy: false,
                });
                notify(`Sorted ${res.data.moved} files into Strong / Mid / Weak`, 'success');
              } catch (err: any) {
                notify(`Could not sort the files. ${err?.response?.data?.detail ?? err.message}`, 'error');
              }
            }}
          >
            Sort files
          </Button>
        )}

        {/* Scan mode. Active state is a luminance step, not a hue — the warm
            colour belongs to the photographer's marks. */}
        {!isGrading && (
          <Button
            variant={scanMode ? 'solid' : 'quiet'}
            onClick={() => setScanMode(v => !v)}
            title={scanMode
              ? 'Scan pass: a quick look at every frame, technical scoring skipped. Click for the full grade.'
              : 'Full grade: quality scoring on every frame. Click for the faster scan pass.'}
            icon={<Zap size={11} fill={scanMode ? 'currentColor' : 'none'}/>}
          >
            Scan
          </Button>
        )}

        {/* Grade. The primary action, but still not accent-coloured: emphasis
            comes from the filled surface and its position at the end of the bar.
            The old pulse animation is gone — a control that throbs at you while
            you are trying to look at photographs is noise. */}
        {isGrading ? (
          <div className="flex h-6 min-w-[180px] shrink-0 items-center gap-2 rounded-sm border border-line-strong bg-raised px-2">
            <div className="h-px flex-1 overflow-hidden rounded-sm bg-well" style={{ height: 3 }}>
              <div
                className="h-full bg-ink-3 transition-[width] duration-slow ease"
                style={{ width: `${Math.max(2, gradeProgress * 100)}%` }}
              />
            </div>
            <span className="t-num shrink-0 text-xs text-ink">
              {Math.round(gradeProgress * 100)}%
            </span>
            {gradeEtaSecs !== null && gradeEtaSecs > 3 && (
              <span className="t-num shrink-0 text-xs text-ink-3">
                {gradeEtaSecs >= 60 ? `${Math.floor(gradeEtaSecs / 60)}m ${gradeEtaSecs % 60}s` : `${gradeEtaSecs}s`}
              </span>
            )}
          </div>
        ) : (
          <Button
            variant="solid"
            onClick={() => handleGrade(true, false)}
            title="Grade every image, replacing existing scores"
            icon={scanMode ? <Zap size={12} fill="currentColor"/> : <Sparkles size={12}/>}
          >
            {isDone ? (scanMode ? 'Re-scan' : 'Re-grade') : (scanMode ? 'Scan' : 'Grade')}
          </Button>
        )}
      </header>

      {/* Progress. A plain ink bar — no gradient. A two-stop gradient sweeping
          across a progress bar is decoration that says nothing the width isn't
          already saying, and it sits directly above the photographs. */}
      <div className="shrink-0"
        role={isGrading ? "progressbar" : undefined}
        aria-label={isGrading ? "Grading progress" : undefined}
        aria-valuenow={isGrading ? Math.round(gradeProgress * 100) : undefined}
        aria-valuemin={isGrading ? 0 : undefined}
        aria-valuemax={isGrading ? 100 : undefined}
        aria-valuetext={isGrading && gradeDesc ? gradeDesc : undefined}>
        <div className="relative overflow-hidden bg-line" style={{ height: 'var(--rule)' }}>
          {listLoading && (
            <div className="absolute top-0 h-full animate-sweep bg-ink-3"/>
          )}
          {!listLoading && isGrading && (
            <div className="h-full bg-ink-2 transition-[width] duration-slow ease"
                 style={{ width: `${Math.max(4, gradeProgress * 100)}%` }}/>
          )}
          {!listLoading && !isGrading && isDone && (
            <div className="h-full w-full bg-grade-strong"/>
          )}
        </div>
        {isGrading && gradeDesc && (() => {
          // Surface the live photo counter (e.g. "44/100") that the backend
          // sends in gradeDesc — toSlogan() rewrites the message into a slogan
          // and would otherwise drop it. Keyed by slogan so the slogan fades in
          // on change while the counter updates in place per photo.
          const _count = (gradeDesc.match(/\d+\s*\/\s*\d+/) || [])[0] || '';
          return (
            <div key={toSlogan(gradeDesc)} style={{ padding:'3px 14px 4px', fontSize:'var(--text-xs)', color:T.ink3, fontStyle:'italic', borderBottom:`1px solid ${T.line}`, animation:'fadeIn .4s cubic-bezier(.2,0,0,1)', display:'flex', gap:8, alignItems:'baseline', justifyContent:'space-between' }}>
              <span>{toSlogan(gradeDesc)}</span>
              <span style={{ display:'flex', gap:8, alignItems:'baseline', flexShrink:0 }}>
                {/* Quality tier actually used for this run. The app picks the
                    best encoder that fits available memory, so this can differ
                    run to run — showing it explains why speed/results vary. */}
                {gradeQuality && (
                  <span title={`Analysis quality for this run: ${gradeQuality}`}
                        style={{ fontStyle:'normal', fontSize:'var(--text-xs)', letterSpacing:.4, textTransform:'uppercase',
                                 color:T.ink2, border:`1px solid ${T.line}`, borderRadius:'var(--r-sm)',
                                 padding:'1px 5px', fontWeight:700 }}>
                    {gradeQuality}
                  </span>
                )}
                {_count && <span style={{ fontStyle:'normal', fontVariantNumeric:'tabular-nums', color:T.ink2, fontWeight:700 }}>{_count}</span>}
              </span>
            </div>
          );
        })()}
        {/* Warm-up transparency — ambient status while background work initialises
            (thumbnail prewarm, model load, niche detect). Hidden during grading,
            which has its own slogan above. */}
        {!isGrading && (() => {
          const warmMsg =
            listLoading ? 'Generating fast-scroll thumbnails…' :
            (graderStatus?.qwen_loading || graderStatus?.warmup_running) ? 'Waking up models…' :
            nicheDetecting ? 'Detecting ideal niche…' :
            null;
          if (!warmMsg) return null;
          return (
            <div style={{ padding:'3px 14px 4px', fontSize:'var(--text-xs)', color:T.ink3, borderBottom:`1px solid ${T.line}`, display:'flex', gap:7, alignItems:'center', animation:'fadeIn .4s cubic-bezier(.2,0,0,1)' }}>
              <div style={{ width:9, height:9, borderRadius:'var(--r-round)', border:'2px solid currentColor', borderTopColor:'transparent', animation:'spin .8s linear infinite' }}/>
              <span>{warmMsg}</span>
            </div>
          );
        })()}
      </div>

      {/* ── Rating filter ──────────────────────────────────────────
       * Star ratings are the photographer's own judgement, so this is one of
       * the few bars allowed to show the warm mark colour. */}
      {mainTab === 'gallery' && isDone && (
        <div className="flex h-8 shrink-0 items-center gap-2 border-b border-line bg-surface px-3">
          <span className="t-label shrink-0">Rating</span>
          <div className="flex gap-1">
            {[1,2,3,4,5].map(n => {
              const active = filterStars === n;
              return (
                <button key={n} onClick={() => setFilterStars(active ? null : n)}
                  title={`${n} star${n > 1 ? 's' : ''}`}
                  className={cn(
                    'flex cursor-pointer items-center gap-1 rounded-sm border px-2 py-px',
                    'transition-colors duration-fast ease',
                    active
                      ? 'border-mark bg-raised'
                      : 'border-line-strong bg-transparent hover:bg-raised',
                  )}>
                  <span className="flex gap-px">
                    {[1,2,3,4,5].map(s => (
                      <svg key={s} width="8" height="8" viewBox="0 0 24 24" strokeWidth="2"
                        fill={s <= n ? T.mark : 'none'}
                        stroke={s <= n ? T.mark : T.ink4}>
                        <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
                      </svg>
                    ))}
                  </span>
                  <span className={cn('t-num min-w-2 text-center text-xs',
                                      active ? 'text-mark-ink' : 'text-ink-3')}>
                    {starCounts[n]}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="h-4 w-px shrink-0 bg-line-strong"/>

          <Button size="sm" variant={filterStars === 0 ? 'solid' : 'quiet'}
            onClick={() => setFilterStars(filterStars === 0 ? null : 0)}>
            Unrated <span className="t-num ml-1 opacity-70">{starCounts[0]}</span>
          </Button>

          {filterStars !== null && (
            <Button size="sm" variant="quiet" onClick={() => setFilterStars(null)}>Clear</Button>
          )}

          {redacted.size > 0 && (
            <>
              <div className="h-4 w-px shrink-0 bg-line-strong"/>
              <Button size="sm" variant={showDuplicates ? 'solid' : 'quiet'}
                onClick={() => setShowDuplicates(v => !v)}
                title={showDuplicates ? 'Hide duplicate shots' : 'Show duplicate shots'}
                icon={<Copy size={10}/>}>
                Dupes <span className="t-num ml-1 opacity-70">{redacted.size}</span>
              </Button>
            </>
          )}

          <span className="ml-auto text-xs text-ink-3">
            <span className="t-num">{filteredPhotos.length}</span> shown
          </span>
        </div>
      )}

      {/* ── Body ───────────────────────────────────────────────── */}
      {mainTab === 'gallery' ? (
        <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', minHeight:0 }}>

          {/* Middle row: grid view OR loupe (preview + right panel) */}
          <div style={{ flex:1, display:'flex', minHeight:0, overflow:'hidden' }}>

            {loupeMode === 'grid' && photos.length > 0 && (
              <GridView
                photos={filteredPhotos}
                selId={selId}
                onSelect={id => { setSelId(id); if (isDone) setLoupeMode('loupe'); }}
                usedPaths={allUsedPaths}
                selectMode={selectMode}
                setSelectMode={setSelectMode}
                selectedIds={selectedIds}
                setSelectedIds={setSelectedIds}
                onCreateSequence={handleCreateFromSelection}
                onAutoSequence={handleGenerate}
              />
            )}

            {(loupeMode === 'loupe' || photos.length === 0) && (<>

            {/* Center preview */}
            {/* The stage. `bg-well` is the one place near-black is correct —
                it sits directly behind a photograph, where a lighter surround
                would wash out the image being judged. */}
            <div className="relative flex min-h-0 min-w-0 flex-1 items-center justify-center overflow-hidden bg-well">
              {photos.length === 0 ? (
                <div className="flex flex-col items-center gap-3">
                  <button
                    onClick={openBrowser}
                    className={cn(
                      'flex cursor-pointer flex-col items-center gap-4 rounded-md border-2 border-dashed',
                      'bg-transparent px-8 py-8 transition-colors duration-slow ease',
                      dragOver ? 'border-ink text-ink' : 'border-line-strong text-ink-3 hover:border-ink-4',
                    )}>
                    <FolderOpen size={40} strokeWidth={1.25} className="text-current"/>
                    <span className={cn('text-lg', dragOver ? 'text-ink' : 'text-ink-2')}>
                      Drop a folder of photos here
                    </span>
                    <span className="text-sm text-ink-3">
                      50 to 100 photos is a good first run
                    </span>
                  </button>
                  {catalogBanner && (
                    <div className="flex items-center gap-3 rounded-sm border border-line-strong bg-surface px-4 py-2">
                      <span className="text-sm text-ink-2">Pick up where you left off?</span>
                      <Button variant="solid" size="sm" onClick={handleResume}>Resume</Button>
                      <Button size="sm" variant="quiet"
                        onClick={() => { axios.post(`${API}/api/catalog/clear`); setCatalogBanner(false); }}>
                        Start fresh
                      </Button>
                    </div>
                  )}
                </div>
              ) : sel ? (
                <>
                  {/* Base photo — always rendered; eye overlay crossfades on top */}
                  <img
                    key={sel.path}
                    src={photoUrl(sel.path)}
                    alt=""
                    onLoad={e => setPhotoNatDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
                    style={{ maxWidth:'100%', maxHeight:'100%', objectFit:'contain', display:'block', userSelect:'none',
                      animation:'fadeIn .35s cubic-bezier(.2,0,0,1)',
                      outline: selectedIds.has(selId ?? '') ? `3px solid ${T.mark}` : 'none',
                      outlineOffset:'-3px', transition:'outline .22s ease',
                    }}
                  />
                  {/* Eye overlay — crossfades in when showEyeOverlay is true */}
                  {sel.eye_overlay_url && (
                    <img
                      key={`overlay-${sel.path}`}
                      src={`${API}${sel.eye_overlay_url}`}
                      alt="judge overlay"
                      style={{ position:'absolute', inset:0, width:'100%', height:'100%',
                        objectFit:'contain', display:'block', pointerEvents:'none',
                        opacity: showEyeOverlay ? 1 : 0,
                        transition:'opacity .35s ease-in-out',
                      }}
                    />
                  )}
                  {/* Floating Judge's Critique toggle button */}
                  {sel.eye_overlay_url && (
                    <button
                      onClick={() => setShowEyeOverlay(v => !v)}
                      title={showEyeOverlay ? "Hide judge's critique" : "Show judge's critique"}
                      style={{
                        position:'absolute', top:10, right:10, zIndex:20,
                        width:34, height:34, borderRadius:'var(--r-md)',
                        display:'flex', alignItems:'center', justifyContent:'center',
                        background: showEyeOverlay ? T.raisedHover : T.well,
                        border: `1px solid ${showEyeOverlay ? T.ink : T.lineStrong}`,
                        backdropFilter:'blur(8px)',
                        cursor:'pointer',
                        transition:'background .2s ease, border-color .2s ease, box-shadow .2s ease',
                        boxShadow: showEyeOverlay ? `0 0 0 2px ${T.lineStrong}` : 'none',
                        color: showEyeOverlay ? T.ink : T.ink2,
                      }}
                      onMouseEnter={e => { if (!showEyeOverlay) (e.currentTarget as HTMLButtonElement).style.background = T.raisedHover; }}
                      onMouseLeave={e => { if (!showEyeOverlay) (e.currentTarget as HTMLButtonElement).style.background = T.well; }}
                    >
                      {/* Eye icon */}
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                        <circle cx="12" cy="12" r="3"/>
                        {showEyeOverlay && <line x1="2" y1="2" x2="22" y2="22" strokeWidth="2"/>}
                      </svg>
                    </button>
                  )}
                  {(showHeatmap || critTrigger === 'blur' || critTrigger === 'heatmap') && heatmapB64 && (
                    <img
                      src={`data:image/png;base64,${heatmapB64}`}
                      alt=""
                      style={{ position:'absolute', top:0, left:0, width:'100%', height:'100%', objectFit:'contain', mixBlendMode:'multiply', pointerEvents:'none', transition:'opacity .2s ease', opacity: critTrigger ? 0.85 : 0.7 }}
                    />
                  )}
                  {critTrigger === 'grid' && (
                    <svg style={{ position:'absolute', top:0, left:0, width:'100%', height:'100%', pointerEvents:'none' }} xmlns="http://www.w3.org/2000/svg">
                      <line x1="33.33%" y1="0%" x2="33.33%" y2="100%" stroke={T.ink} strokeOpacity={0.55} strokeWidth="1"/>
                      <line x1="66.66%" y1="0%" x2="66.66%" y2="100%" stroke={T.ink} strokeOpacity={0.55} strokeWidth="1"/>
                      <line x1="0%" y1="33.33%" x2="100%" y2="33.33%" stroke={T.ink} strokeOpacity={0.55} strokeWidth="1"/>
                      <line x1="0%" y1="66.66%" x2="100%" y2="66.66%" stroke={T.ink} strokeOpacity={0.55} strokeWidth="1"/>
                    </svg>
                  )}
                  {/* Criteria overlay — shown when eye/audit mode is active */}
                  {isAuditModeActive && isGraded && photoNatDims && (() => {
                    const _bd_vlm = sel?.breakdown as any;
                    const _bboxes = (_bd_vlm?.vlm_bboxes as Array<{label:string;bbox_2d:number[]}>) ?? [];
                    const W = photoNatDims.w, H = photoNatDims.h;
                    const sw  = Math.max(3, W * 0.003);

                    // Aspect scores
                    const _ASPECT_KEYS = ['Composition','Lighting','Narrative','Atmosphere','Geometry','Technical','Human/Culture'];
                    const _aspects = _ASPECT_KEYS
                      .map(k => [k === 'Human/Culture' ? 'Human' : k, _bd_vlm?.[k]] as [string, number])
                      .filter(([, v]) => typeof v === 'number' && v > 0.05);
                    const _strongA = [..._aspects].sort((a,b) => (b[1] as number)-(a[1] as number)).find(([,v]) => (v as number) >= 0.55) ?? null;
                    const _weakA   = [..._aspects].sort((a,b) => (a[1] as number)-(b[1] as number)).find(([,v]) => (v as number) < 0.55) ?? null;

                    // Typography — scales with image resolution
                    const nameFs   = Math.max(26, W * 0.026);
                    const labelFs  = Math.max(14, W * 0.015);
                    const edge     = Math.max(14, W * 0.020);
                    const ulThick  = Math.max(2.5, W * 0.0028);
                    const weakCol  = T.alarmWarn;

                    // Approximate text width for underline sizing
                    const _tw = (t: string, fs: number) => t.length * fs * 0.60;

                    return (
                      <svg
                        style={{ position:'absolute', inset:0, width:'100%', height:'100%',
                          pointerEvents:'none', zIndex:5,
                          animation:'fadeIn .3s cubic-bezier(.2,0,0,1)' }}
                        viewBox={`0 0 ${W} ${H}`}
                        preserveAspectRatio="xMidYMid meet"
                      >
                        {/* ── Strongest aspect — top-left ─────────────────── */}
                        {_strongA && (() => {
                          const name = (_strongA[0] as string).toUpperCase();
                          const uw   = _tw('✓ ' + name, nameFs);
                          const ty   = edge + nameFs;
                          return (
                            <g style={{ animation:'fadeIn .5s .06s both' }}>
                              <text x={edge} y={ty}
                                fill={T.ink} fontSize={nameFs} fontWeight="700"
                                fontFamily="'SF Mono',ui-monospace,monospace"
                                stroke={T.well} strokeOpacity={0.75} strokeWidth={sw*1.8} paintOrder="stroke fill">
                                <tspan fill={T.gradeStrong} fontWeight="800">{'✓ '}</tspan>{name}
                              </text>
                              <line x1={edge} y1={ty + nameFs*0.20}
                                    x2={edge + uw} y2={ty + nameFs*0.20}
                                stroke={T.gradeStrong} strokeWidth={ulThick} strokeLinecap="round"/>
                              <text x={edge} y={ty + nameFs*0.20 + labelFs*1.5}
                                fill={T.gradeStrong} fontSize={labelFs} fontWeight="600"
                                fontFamily="'SF Mono',ui-monospace,monospace" opacity={0.75}
                                stroke={T.well} strokeOpacity={0.65} strokeWidth={sw*1.4} paintOrder="stroke fill">
                                STRONGEST ASPECT
                              </text>
                            </g>
                          );
                        })()}

                        {/* ── Weakest aspect — top-right ──────────────────── */}
                        {_weakA && (() => {
                          const name = (_weakA[0] as string).toUpperCase();
                          const uw   = _tw(name + ' ↑', nameFs);
                          const ty   = edge + nameFs;
                          return (
                            <g style={{ animation:'fadeIn .5s .16s both' }}>
                              <text x={W - edge} y={ty}
                                textAnchor="end"
                                fill={T.ink} fontSize={nameFs} fontWeight="700"
                                fontFamily="'SF Mono',ui-monospace,monospace"
                                stroke={T.well} strokeOpacity={0.75} strokeWidth={sw*1.8} paintOrder="stroke fill">
                                {name}<tspan fill={weakCol} fontWeight="800">{' ↑'}</tspan>
                              </text>
                              <line x1={W - edge - uw} y1={ty + nameFs*0.20}
                                    x2={W - edge}       y2={ty + nameFs*0.20}
                                stroke={weakCol} strokeWidth={ulThick} strokeLinecap="round"/>
                              <text x={W - edge} y={ty + nameFs*0.20 + labelFs*1.5}
                                textAnchor="end"
                                fill={weakCol} fontSize={labelFs} fontWeight="600"
                                fontFamily="'SF Mono',ui-monospace,monospace" opacity={0.75}
                                stroke={T.well} strokeOpacity={0.65} strokeWidth={sw*1.4} paintOrder="stroke fill">
                                NEEDS MOST WORK
                              </text>
                            </g>
                          );
                        })()}

                        {/* ── Spatial bboxes from pipeline — teacher callouts ─── */}
                        {_bboxes.map((b, bi) => {
                          if (!b.bbox_2d?.length) return null;
                          // Subject-less fallback: render a rule-of-thirds grid (lines)
                          // instead of a centered box + chip, which read as a black blob.
                          if (b.label === 'compositional_center') {
                            const gc = T.ink2;
                            const dash = `${sw*3} ${sw*2}`;
                            return (
                              <g key={bi} style={{ animation:`fadeIn .4s ${bi*0.08}s both` }}>
                                {[1,2].map(k => (
                                  <line key={'v'+k} x1={W*k/3} y1={0} x2={W*k/3} y2={H}
                                    stroke={gc} strokeWidth={sw*0.7} strokeDasharray={dash} opacity={0.7}/>
                                ))}
                                {[1,2].map(k => (
                                  <line key={'h'+k} x1={0} y1={H*k/3} x2={W} y2={H*k/3}
                                    stroke={gc} strokeWidth={sw*0.7} strokeDasharray={dash} opacity={0.7}/>
                                ))}
                                {[1,2].flatMap(cx => [1,2].map(cy => (
                                  <circle key={`p${cx}${cy}`} cx={W*cx/3} cy={H*cy/3} r={sw*1.8}
                                    fill="none" stroke={gc} strokeWidth={sw*0.9} opacity={0.9}/>
                                )))}
                              </g>
                            );
                          }
                          const [x1, y1, x2, y2] = b.bbox_2d;
                          const bw  = Math.max(x2-x1, sw*4), bh = Math.max(y2-y1, sw*4);
                          const guide = regionGuide(b.label);
                          const col   = tierColor(guide.tier);
                          const title = `${tierIcon(guide.tier)}  ${guide.title.toUpperCase()}`;
                          const tip   = guide.tip;
                          // Two-line chip: bold title + smaller coaching tip
                          const titleFs = Math.max(15, W * 0.0145);
                          const tipFs   = Math.max(12, W * 0.0118);
                          const pad     = sw * 2.6;
                          const titleW  = _tw(title, titleFs);
                          const tipW    = tip ? _tw(tip, tipFs) : 0;
                          const chipW   = Math.max(titleW, tipW) + pad * 2;
                          const chipH   = tip ? titleFs*1.35 + tipFs*1.45 + pad*1.4
                                              : titleFs*1.35 + pad*1.2;
                          const chipY   = y1 >= chipH + sw*3 ? y1 - chipH - sw : y2 + sw;
                          const chipX   = Math.min(Math.max(x1, sw), W - chipW - sw);
                          return (
                            <g key={bi} style={{ animation:`fadeIn .35s ${bi*0.08}s both` }}>
                              {/* Region fill */}
                              <rect x={x1} y={y1} width={bw} height={bh}
                                fill={col+'10'} stroke={col} strokeWidth={sw*0.8}
                                strokeDasharray={`${sw*4} ${sw*2}`} rx={sw*1.5}/>
                              {/* Corner marks */}
                              <path d={`M${x1+sw} ${y1+sw*5} L${x1+sw} ${y1+sw} L${x1+sw*5} ${y1+sw}`}
                                fill="none" stroke={col} strokeWidth={sw*1.3} strokeLinecap="round"/>
                              <path d={`M${x2-sw*5} ${y1+sw} L${x2-sw} ${y1+sw} L${x2-sw} ${y1+sw*5}`}
                                fill="none" stroke={col} strokeWidth={sw*1.3} strokeLinecap="round"/>
                              <path d={`M${x1+sw} ${y2-sw*5} L${x1+sw} ${y2-sw} L${x1+sw*5} ${y2-sw}`}
                                fill="none" stroke={col} strokeWidth={sw*1.3} strokeLinecap="round"/>
                              <path d={`M${x2-sw*5} ${y2-sw} L${x2-sw} ${y2-sw} L${x2-sw} ${y2-sw*5}`}
                                fill="none" stroke={col} strokeWidth={sw*1.3} strokeLinecap="round"/>
                              {/* Teacher chip */}
                              <rect x={chipX} y={chipY} width={chipW} height={chipH}
                                fill={T.well} fillOpacity={0.86} stroke={col} strokeWidth={sw*0.5} rx={sw*1.4}/>
                              {/* Accent bar */}
                              <rect x={chipX} y={chipY} width={sw*1.2} height={chipH}
                                fill={col} rx={sw*0.6}/>
                              <text x={chipX+pad} y={chipY+pad*0.6+titleFs}
                                fill={col} fontSize={titleFs} fontWeight="800"
                                fontFamily="'SF Mono',ui-monospace,monospace">{title}</text>
                              {tip && (
                                <text x={chipX+pad} y={chipY+pad*0.6+titleFs+tipFs*1.4}
                                  fill={T.ink} fillOpacity={0.88} fontSize={tipFs} fontWeight="500"
                                  fontFamily="ui-sans-serif,system-ui,sans-serif">{tip}</text>
                              )}
                            </g>
                          );
                        })}
                      </svg>
                    );
                  })()}
                  {/* ── Critique heatmap — part of Vision Critique (same toggle) ── */}
                  {isAuditModeActive && isGraded && photoNatDims && (() => {
                    const _bd     = sel?.breakdown as any;
                    const _bboxes = (_bd?.vlm_bboxes as Array<{label:string;bbox_2d:number[]}>) ?? [];
                    if (!_bboxes.length) return null;
                    const W = photoNatDims.w, H = photoNatDims.h;
                    const _tierFromVal = (v: number): RegionTier =>
                      v >= 0.6 ? 'strong' : v < 0.4 ? 'fix' : 'refine';
                    const _gradeTier = (g: string): RegionTier =>
                      g?.includes('Strong') ? 'strong' : g?.includes('Weak') ? 'fix' : 'refine';

                    // Heat points. When the only region is the subject-less fallback,
                    // derive one point per graded aspect (placed around the frame,
                    // coloured by that aspect's strength) so even a weak photo shows a
                    // spread of weak/mid/strong points instead of one flat blob.
                    type HeatPt = { cx:number; cy:number; rx:number; ry:number; tier:RegionTier };
                    const _isFallback = _bboxes.length === 1 && _bboxes[0].label === 'compositional_center';
                    let _points: HeatPt[] = [];
                    if (_isFallback) {
                      const POS: Record<string,[number,number]> = {
                        composition:[0.50,0.50], lighting:[0.50,0.24], narrative:[0.28,0.54],
                        technical:[0.74,0.74], 'human/culture':[0.74,0.40], human:[0.74,0.40],
                        geometry:[0.28,0.28], detail:[0.74,0.26], atmosphere:[0.50,0.76],
                        moment:[0.40,0.42], light_quality:[0.50,0.24],
                      };
                      const FALLBACK: [number,number][] = [[0.5,0.5],[0.3,0.3],[0.7,0.3],[0.3,0.72],[0.7,0.72],[0.5,0.24],[0.5,0.78]];
                      let fi = 0;
                      _points = Object.entries(_bd || {})
                        .filter(([k,v]) => typeof v === 'number' && !k.startsWith('_')
                          && !['Aesthetic','Personal','aesthetic','personal','overall_score','score','gemma_score'].includes(k))
                        .map(([k,v]) => {
                          const pos = POS[k.toLowerCase()] ?? FALLBACK[fi++ % FALLBACK.length];
                          return { cx: pos[0]*W, cy: pos[1]*H, rx: W*0.155, ry: H*0.155, tier: _tierFromVal(v as number) };
                        });
                      if (!_points.length) _points = [{ cx:W/2, cy:H/2, rx:W*0.30, ry:H*0.30, tier:_gradeTier(sel?.grade || '') }];
                    } else {
                      _points = _bboxes.filter(b => b.bbox_2d?.length).map(b => {
                        const [x1,y1,x2,y2] = b.bbox_2d;
                        return { cx:(x1+x2)/2, cy:(y1+y2)/2,
                          rx:Math.max((x2-x1)/2*1.45, W*0.06), ry:Math.max((y2-y1)/2*1.45, H*0.06),
                          tier: regionGuide(b.label).tier };
                      });
                    }
                    return (
                      <svg
                        style={{ position:'absolute', inset:0, width:'100%', height:'100%',
                          pointerEvents:'none', zIndex:4,
                          animation:'fadeIn .3s cubic-bezier(.2,0,0,1)' }}
                        viewBox={`0 0 ${W} ${H}`}
                        preserveAspectRatio="xMidYMid meet"
                      >
                        <defs>
                          {_points.map((p, i) => {
                            const c = tierHeat(p.tier);
                            return (
                              <radialGradient key={i} id={`crit-heat-${i}`} cx="50%" cy="50%" r="50%">
                                <stop offset="0%"   stopColor={c} stopOpacity="0.92"/>
                                <stop offset="35%"  stopColor={c} stopOpacity="0.62"/>
                                <stop offset="70%"  stopColor={c} stopOpacity="0.28"/>
                                <stop offset="100%" stopColor={c} stopOpacity="0"/>
                              </radialGradient>
                            );
                          })}
                        </defs>
                        {/* soft heat field */}
                        {_points.map((p, i) => (
                          <ellipse key={`g${i}`} cx={p.cx} cy={p.cy} rx={p.rx} ry={p.ry} fill={`url(#crit-heat-${i})`}/>
                        ))}
                        {/* crisp tier core + halo so each point reads clearly on any image */}
                        {_points.map((p, i) => {
                          const c = tierHeat(p.tier);
                          const r = Math.max(p.rx, p.ry) * 0.16;
                          return (
                            <g key={`c${i}`}>
                              <circle cx={p.cx} cy={p.cy} r={r * 1.9} fill="none"
                                stroke={c} strokeWidth={r * 0.32} opacity={0.55}/>
                              <circle cx={p.cx} cy={p.cy} r={r} fill={c} opacity={0.95}
                                stroke={T.well} strokeOpacity={0.45} strokeWidth={r * 0.18}/>
                            </g>
                          );
                        })}
                      </svg>
                    );
                  })()}
                  {/* Heatmap legend — shows only the tiers actually present, with counts */}
                  {isAuditModeActive && isGraded && ((sel?.breakdown as any)?.vlm_bboxes?.length > 0) && (() => {
                    const _bdL    = sel?.breakdown as any;
                    const _b = (_bdL?.vlm_bboxes as Array<{label:string}>) ?? [];
                    const _tierFromVal = (v: number): RegionTier =>
                      v >= 0.6 ? 'strong' : v < 0.4 ? 'fix' : 'refine';
                    const _counts: Record<RegionTier, number> = { strong:0, refine:0, fix:0 };
                    const _isFallbackL = _b.length === 1 && _b[0].label === 'compositional_center';
                    if (_isFallbackL) {
                      // Mirror the aspect-point heatmap: count one tier per graded aspect.
                      Object.entries(_bdL || {})
                        .filter(([k,v]) => typeof v === 'number' && !k.startsWith('_')
                          && !['Aesthetic','Personal','aesthetic','personal','overall_score','score','gemma_score'].includes(k))
                        .forEach(([,v]) => { _counts[_tierFromVal(v as number)]++; });
                    } else {
                      _b.forEach(x => { _counts[regionGuide(x.label).tier]++; });
                    }
                    const _rows: Array<[RegionTier, string]> = [
                      ['strong','Strong'], ['refine','Could be stronger'], ['fix','Needs work'],
                    ];
                    const _present = _rows.filter(([t]) => _counts[t] > 0);
                    if (!_present.length) return null;
                    return (
                      <div style={{ position:'absolute', bottom:16, right:16, zIndex:20,
                        display:'flex', flexDirection:'column', gap:5, padding:'9px 12px',
                        background:T.scrim, border:`1px solid ${T.lineStrong}`,
                        borderRadius:'var(--r-md)', backdropFilter:'blur(10px)', pointerEvents:'none' }}>
                        <div className="t-label" style={{ marginBottom:1 }}>Critique map</div>
                        {_present.map(([t, l]) => {
                          const c = tierHeat(t);
                          return (
                            <div key={t} style={{ display:'flex', alignItems:'center', gap:7 }}>
                              <span style={{ width:11, height:11, borderRadius:'var(--r-round)', flexShrink:0,
                                background:`radial-gradient(circle, ${c} 0%, transparent 100%)`,
                                boxShadow:`0 0 6px ${c}` }}/>
                              <span style={{ fontSize:'var(--text-xs)', color:T.ink }}>{l}</span>
                              <span className="t-num" style={{ fontSize:'var(--text-xs)', color:T.ink3, marginLeft:'auto', paddingLeft:8 }}>{_counts[t]}</span>
                            </div>
                          );
                        })}
                      </div>
                    );
                  })()}
                  <button onClick={() => hasPrev && setSelId(filteredPhotos[selIdx-1].id)} disabled={!hasPrev}
                    style={{ position:'absolute', left:12, top:'50%', transform:'translateY(-50%)', width:34, height:34, borderRadius:'var(--r-round)', display:'flex', alignItems:'center', justifyContent:'center', background:T.scrim, backdropFilter:'blur(12px)', color:hasPrev?T.ink:T.ink3, opacity:hasPrev?1:0, border:`1px solid ${T.lineStrong}`, pointerEvents:hasPrev?'auto':'none', cursor:'pointer', fontSize:'var(--text-md)' }}>‹</button>
                  <button onClick={() => hasNext && setSelId(filteredPhotos[selIdx+1].id)} disabled={!hasNext}
                    style={{ position:'absolute', right:12, top:'50%', transform:'translateY(-50%)', width:34, height:34, borderRadius:'var(--r-round)', display:'flex', alignItems:'center', justifyContent:'center', background:T.scrim, backdropFilter:'blur(12px)', color:hasNext?T.ink:T.ink3, opacity:hasNext?1:0, border:`1px solid ${T.lineStrong}`, pointerEvents:hasNext?'auto':'none', cursor:'pointer', fontSize:'var(--text-md)' }}>›</button>
                  {/* Select toggle */}
                  {selId && (() => {
                    const isSel = selectedIds.has(selId);
                    return (
                      <button onClick={() => setSelectedIds(prev => { const next = new Set(prev); next.has(selId) ? next.delete(selId) : next.add(selId); return next; })}
                        className="absolute bottom-4 left-4 flex cursor-pointer items-center gap-1 rounded-md border px-3 py-1 text-xs transition-colors duration-fast ease"
                        style={{ background:isSel ? T.mark : T.well, borderColor:isSel ? T.mark : T.lineStrong, color:isSel ? T.well : T.ink }}>
                        <span className="flex h-3 w-3 shrink-0 items-center justify-center rounded-sm border"
                              style={{ background:isSel ? T.well : 'transparent', borderColor:isSel ? T.well : T.ink2 }}>
                          {isSel && <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke={T.mark} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20,6 9,17 4,12"/></svg>}
                        </span>
                        {isSel ? 'Selected' : 'Select'}
                      </button>
                    );
                  })()}
                  {/* Floating action bar */}
                  {selectedIds.size > 0 && (
                    <div style={{ position:'absolute', bottom:16, left:'50%', transform:'translateX(-50%) translateX(40px)', display:'flex', alignItems:'center', gap:10, background:T.surface, border:`1px solid ${T.lineStrong}`, borderRadius:'var(--r-md)', padding:'10px 18px', boxShadow:`0 8px 40px ${T.well}`, backdropFilter:'blur(12px)', zIndex:50, whiteSpace:'nowrap', animation:'slideUp .3s cubic-bezier(.2,0,0,1)' }}>
                      <span style={{ fontSize:'var(--text-sm)', fontWeight:700, color:T.ink }}>{selectedIds.size} selected</span>
                      <div style={{ width:1, height:16, background:T.lineStrong }}/>
                      <Button variant="solid" onClick={handleCreateFromSelection} icon={<Layers size={11}/>}>
                        Start Sequence
                      </Button>
                      <button onClick={handleGenerate}
                        style={{ display:'flex', alignItems:'center', gap:6, padding:'6px 14px', borderRadius:'var(--r-md)', background:T.raised, border:`1px solid ${T.lineStrong}`, color:T.ink2, fontSize:'var(--text-sm)', fontWeight:600, cursor:'pointer' }}>
                        <RefreshCw size={11}/> Auto
                      </button>
                    </div>
                  )}
                </>
              ) : null}
            </div>

            {/* Resize handle */}
            {photos.length > 0 && (
            <div
              onMouseDown={onResizeDown}
              style={{ width:3, cursor:'col-resize', flexShrink:0, background:'transparent', transition:'background .25s ease' }}
              onMouseEnter={e => (e.currentTarget.style.background = T.raisedHover)}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            />
            )}

            {/* Right panel */}
            {photos.length > 0 && <div style={{ width:rightW, flexShrink:0, background:T.surface, borderLeft:`1px solid ${T.line}`, display:'flex', flexDirection:'column', overflow:'hidden' }}>

              {/* Thumbnail */}
              {sel && (
                <div style={{ flexShrink:0, position:'relative', aspectRatio:'3/2', background:T.ground, overflow:'hidden' }}>
                  <img key={sel.path} src={thumbUrl(sel.path)} alt=""
                    style={{ width:'100%', height:'100%', objectFit:'cover', display:'block', animation:'fadeIn .32s cubic-bezier(.2,0,0,1)' }}/>
                  {isGraded && (
                    <div style={{ position:'absolute', inset:0, background:`linear-gradient(to top,${T.scrim} 0%,transparent 55%)`, display:'flex', alignItems:'flex-end', padding:'10px 12px' }}>
                      <div style={{ display:'flex', alignItems:'center', gap:6, background:T.scrim, backdropFilter:'blur(8px)', borderRadius:'var(--r-md)', padding:'6px 12px', border:`1px solid ${gc(sel.grade)}` }}>
                        <div style={{ width:8, height:8, borderRadius:'var(--r-round)', background:gc(sel.grade), flexShrink:0 }}/>
                        <span style={{ fontSize:'var(--text-md)', fontWeight:700, color:T.ink }}>{gradeLabel(sel.grade)}</span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Filename + copy + stars */}
              {sel && (
                <div style={{ flexShrink:0, padding:'10px 14px', borderBottom:`1px solid ${T.line}` }}>
                  <div style={{ display:'flex', alignItems:'center', gap:6, marginBottom:6 }}>
                    <span style={{ flex:1, fontSize:'var(--text-sm)', fontWeight:600, color:T.ink, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                      {sel.path.split(/[\\/]/).pop()}
                    </span>
                    <button onClick={handleCopyPath} title="Copy path"
                      style={{ display:'flex', alignItems:'center', gap:4, padding:'4px 7px', borderRadius:'var(--r-sm)', background:copied ? T.raisedHover : T.raised, border:`1px solid ${T.lineStrong}`, color:copied ? T.gradeStrong : T.ink3, fontSize:'var(--text-xs)', cursor:'pointer', transition:'all .25s cubic-bezier(.2,0,0,1)' }}>
                      <Copy size={10}/>{copied ? 'Copied!' : ''}
                    </button>
                  </div>
                  <StarRating stars={sel.stars ?? 0} size={22} onSet={n => handleSetStars(sel.id, n)}/>
                  {/* Grade display — read-only */}
                  {isDone && (
                    <div style={{ display:'flex', gap:4, marginTop:8 }}>
                      {(['Strong ✅','Mid ⚠️','Weak ❌'] as const).map(g => {
                        const _sc = sel.score ?? 0;
                        const derivedGrade = _sc >= 0.60 ? 'Strong ✅' : _sc >= 0.41 ? 'Mid ⚠️' : 'Weak ❌';
                        const isActive = derivedGrade === g;
                        const col = g.includes('Strong') ? T.gradeStrong : g.includes('Mid') ? T.ink2 : T.gradeWeak;
                        return (
                          <div key={g}
                            style={{ flex:1, padding:'3px 0', borderRadius:'var(--r-sm)', fontSize:'var(--text-xs)', fontWeight:700,
                              textAlign:'center', userSelect:'none', pointerEvents:'none',
                              background: isActive ? T.raisedHover : 'transparent',
                              border: `1px solid ${isActive ? col : T.lineStrong}`,
                              color: isActive ? col : T.ink3 }}>
                            {gradeLabel(g)}
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {/* Duplicate badge — shown when this photo is the best in a group */}
                  {isDone && sel.cluster_id >= 0 && (sel.sim_flag || '').startsWith('★') && (() => {
                    const m = (sel.sim_flag as string).match(/Best of (\d+)/);
                    const count = m ? parseInt(m[1]) : 2;
                    return (
                      <button
                        onClick={() => setMainTab('duplicates')}
                        style={{ display:'flex', alignItems:'center', gap:5, marginTop:8, width:'100%',
                          padding:'5px 10px', borderRadius:'var(--r-md)', cursor:'pointer',
                          background:T.raised, border:`1px solid ${T.lineStrong}`,
                          color:T.ink2, fontSize:'var(--text-xs)', fontWeight:600, transition:'all .22s cubic-bezier(.2,0,0,1)' }}>
                        <Layers size={10} style={{ flexShrink:0 }}/>
                        Best of {count} similar shots — view duplicates
                      </button>
                    );
                  })()}
                </div>
              )}

              {/* Panel tabs. The active tab is marked by an ink underline and a
                  luminance step, not the accent — warm is reserved for marks. */}
              {sel && (
                <div className="flex shrink-0 border-b border-line" role="tablist">
                  {(isDone
                    ? [['breakdown','Breakdown'],['analysis','Analysis'],['exif','EXIF']]
                    : [['exif','EXIF']]
                  ).map(([id, label]) => (
                    <button key={id}
                      role="tab"
                      aria-selected={infoTab === id}
                      onClick={() => setInfoTab(id as any)}
                      className={cn(
                        '-mb-px h-6 flex-1 cursor-pointer border-0 border-b-2 bg-transparent',
                        'text-sm font-medium transition-colors duration-fast ease',
                        infoTab === id
                          ? 'border-ink text-ink'
                          : 'border-transparent text-ink-3 hover:bg-raised hover:text-ink-2',
                      )}>
                      {label}
                    </button>
                  ))}
                </div>
              )}

              {/* Panel body */}
              <div className="flex-1 overflow-y-auto p-4">
                {infoTab === 'exif' && (
                  sel
                    ? <ExifPanel exif={sel.exif ?? {}}/>
                    : null
                )}
                {infoTab === 'analysis' && (
                  sel && isGraded ? (() => {
                    const bd = sel.breakdown ?? {};
                    // Verified photos: use stored 7B chain-of-thought (richer, model-generated).
                    // All others: always regenerate from aspect scores so moody-aware text
                    // is applied fresh — bypasses any old penalizing text stored in the DB.
                    const rl = (sel.is_verified && sel.reasoning_log)
                      ? sel.reasoning_log
                      : (Object.keys(bd).length > 0
                          ? buildReasoningFromBreakdown(sel.score, sel.grade, bd)
                          : sel.reasoning_log || '');
                    const rlines   = rl.split('\n');
                    const header   = rlines[0] ?? '';
                    const verdict  = rlines[1] ?? '';
                    const footer   = rlines.find(l => l.trimStart().startsWith('Best:')) ?? '';
                    const obsLines = rlines.slice(3).filter(l => l && !l.trimStart().startsWith('Best:'));
                    const tierWord = header.split(/\s+/)[0] ?? '';
                    const scorePct = header.match(/(\d+)%/)?.[1] ?? '';
                    const gradeCol = gc(sel.grade ?? '');
                    const _handleReveal = async () => {
                      if (!sel?.path) return;
                      setDeepCritiqueLoading(true);
                      try {
                        const r = await axios.post(`${API}/api/critique/details`, {
                          image_path: sel.path,
                          mode: (sel as any).preset || 'story',
                        });
                        setDeepCritique(r.data);
                      } catch {
                        setDeepCritique({ narrative_arc: 'The writing engine isn’t running, so no critique could be written. Grading and scores are unaffected.', geometry_composition: '' });
                      } finally {
                        setDeepCritiqueLoading(false);
                      }
                    };
                    return (
                      <div style={{ display:'flex', flexDirection:'column', gap:14, animation:'fadeIn .32s cubic-bezier(.2,0,0,1)' }}>
                        {/* Draw-on-image toggle + deep critique trigger */}
                        <button
                          onClick={() => {
                            const next = !isAuditModeActive;
                            setIsAuditModeActive(next);
                            if (next && !deepCritique) _handleReveal();
                          }}
                          title={isAuditModeActive ? 'Hide annotation overlay' : 'Show critique overlay and narrative analysis'}
                          style={{ display:'flex', alignItems:'center', gap:7, padding:'7px 13px',
                            borderRadius:'var(--r-md)', alignSelf:'flex-start', cursor:'pointer',
                            fontWeight:700, fontSize:'var(--text-xs)', letterSpacing:'.03em',
                            border:`1px solid ${isAuditModeActive ? T.lineStrong : T.line}`,
                            background: isAuditModeActive ? T.raisedHover : T.raised,
                            color: isAuditModeActive ? T.ink : T.ink2,
                            transition:'all .2s cubic-bezier(.2,0,0,1)' }}>
                          {isAuditModeActive ? <EyeOff size={12}/> : <Eye size={12}/>}
                          {isAuditModeActive ? 'Hide Critique' : 'Vision Critique'}
                          {deepCritiqueLoading && (
                            <span style={{ width:8, height:8, borderRadius:'var(--r-round)',
                              border:'1.5px solid currentColor', borderTopColor:'transparent',
                              animation:'spin .8s linear infinite', display:'inline-block' }}/>
                          )}
                          {!deepCritiqueLoading && isAuditModeActive && reasoningOverlayUrl && (
                            <span style={{ width:5, height:5, borderRadius:'var(--r-round)',
                              background:T.gradeStrong, flexShrink:0 }}/>
                          )}
                        </button>
                        {/* VERIFIED badge */}
                        {sel.is_verified && (
                          <div style={{ display:'inline-flex', alignItems:'center', gap:5, padding:'4px 10px', borderRadius:'var(--r-sm)', background:T.raised, border:`1px solid ${T.gradeStrong}`, alignSelf:'flex-start' }}>
                            <div style={{ width:6, height:6, borderRadius:'var(--r-round)', background:T.gradeStrong }}/>
                            <span style={{ fontSize:'var(--text-xs)', fontWeight:700, letterSpacing:'.08em', color:T.gradeStrong }}>VERIFIED</span>
                          </div>
                        )}
                        {/* Tier label */}
                        {tierWord && (
                          <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                            <div style={{ width:10, height:10, borderRadius:'var(--r-round)', background:gradeCol, flexShrink:0 }}/>
                            <span style={{ fontSize:'var(--text-lg)', fontWeight:800, letterSpacing:'.08em', color:gradeCol }}>{tierWord.toUpperCase()}</span>
                          </div>
                        )}
                        {/* Verdict */}
                        {verdict && (
                          <p style={{ fontSize:'var(--text-sm)', color:T.ink2, lineHeight:1.7, margin:0, fontStyle:'italic' }}>{verdict}</p>
                        )}
                        {/* Per-aspect observations */}
                        {obsLines.length > 0 && (
                          <div style={{ display:'flex', flexDirection:'column', gap:1,
                            borderRadius:'var(--r-md)', overflow:'hidden', border:`1px solid ${T.line}` }}>
                            {obsLines.map((line, idx) => {
                              const colon = line.indexOf(':');
                              const label = colon > 0 ? line.slice(0, colon).trim() : '';
                              const note  = colon > 0 ? line.slice(colon + 1).trim() : line;
                              const bdKey = label === 'Moment' ? 'Narrative'
                                          : label === 'Human'  ? 'Human/Culture'
                                          : label;
                              const v    = typeof bd[bdKey] === 'number' ? bd[bdKey] as number : null;
                              const vpct = v !== null ? Math.round(v * 100) : null;
                              const bc   = v === null ? T.ink3
                                         : v >= 0.6  ? T.gradeStrong
                                         : v >= 0.41 ? T.ink2 : T.gradeWeak;
                              const isLast = idx === obsLines.length - 1;
                              return (
                                <div key={idx} style={{ padding:'10px 13px',
                                  background: idx % 2 === 0 ? T.raised : T.ground,
                                  borderBottom: isLast ? 'none' : `1px solid ${T.line}` }}>
                                  <div style={{ display:'flex', justifyContent:'space-between',
                                    alignItems:'center', marginBottom: v !== null ? 5 : 0 }}>
                                    {label && (
                                      <span style={{ fontSize:'var(--text-xs)', fontWeight:700,
                                        letterSpacing:'.08em', color:bc }}>{label.toUpperCase()}</span>
                                    )}
                                    {vpct !== null && (
                                      <span style={{ fontSize:'var(--text-xs)', fontWeight:600, letterSpacing:'.05em', textTransform:'uppercase',
                                        color:bc }}>{vpct >= 60 ? 'Strong' : vpct >= 41 ? 'Mid' : 'Weak'}</span>
                                    )}
                                  </div>
                                  {v !== null && (
                                    <div style={{ height:2, background:T.ground, borderRadius:'var(--r-sm)',
                                      overflow:'hidden', marginBottom:6 }}>
                                      <div style={{ width:`${vpct}%`, height:'100%',
                                        background:bc, borderRadius:'var(--r-sm)',
                                        transition:'width .5s cubic-bezier(.2,0,0,1)' }}/>
                                    </div>
                                  )}
                                  <p style={{ fontSize:'var(--text-xs)', color:T.ink2, margin:0, lineHeight:1.6 }}>{note}</p>
                                </div>
                              );
                            })}
                          </div>
                        )}
                        {/* Best / Weakest */}
                        {footer && (
                          <p style={{ fontSize:'var(--text-xs)', color:T.ink3, margin:0, letterSpacing:'.02em' }}>{footer.trim()}</p>
                        )}
                        {/* Vision Critique — fast-scan bboxes + on-demand deep text */}
                        {(() => {
                          const _bdv    = bd as any;
                          const _vbboxes = (_bdv.vlm_bboxes as Array<{label:string;bbox_2d:number[];justification?:string}>) || [];
                          const _qwenCritique = (_bdv._critique as string) || '';
                          if (!_vbboxes.length && !deepCritique && !_qwenCritique) return null;

                          const _dnarr = deepCritique?.narrative_arc        || '';
                          const _dgeo  = deepCritique?.geometry_composition || '';
                          const _hasDeep = Boolean(_dnarr || _dgeo);

                          return (
                            <div style={{ display:'flex', flexDirection:'column', gap:10,
                              paddingTop:10, borderTop:`1px solid ${T.line}` }}>
                              <div style={{ display:'flex', alignItems:'center' }}>
                                <span className="t-label !text-ink-2">
                                  VISION CRITIQUE
                                </span>
                              </div>
                              {_dnarr && (
                                <div style={{ animation:'fadeIn .4s cubic-bezier(.2,0,0,1)' }}>
                                  <span style={{ fontSize:'var(--text-xs)', fontWeight:700, letterSpacing:'.08em', color:T.ink3 }}>NARRATIVE</span>
                                  <p style={{ fontSize:'var(--text-xs)', color:T.ink2, lineHeight:1.65, margin:'4px 0 0' }}>{_dnarr}</p>
                                </div>
                              )}
                              {_dgeo && (
                                <div style={{ animation:'fadeIn .4s cubic-bezier(.2,0,0,1)' }}>
                                  <span style={{ fontSize:'var(--text-xs)', fontWeight:700, letterSpacing:'.08em', color:T.ink3 }}>GEOMETRY</span>
                                  <p style={{ fontSize:'var(--text-xs)', color:T.ink2, lineHeight:1.65, margin:'4px 0 0' }}>{_dgeo}</p>
                                </div>
                              )}
                              {_qwenCritique && !_hasDeep && (
                                <p style={{ fontSize:'var(--text-sm)', color:T.ink2, lineHeight:1.7, margin:0,
                                  fontStyle:'italic', padding:'8px 10px', background:T.raised,
                                  borderRadius:'var(--r-md)', border:`1px solid ${T.line}` }}>
                                  {_qwenCritique}
                                </p>
                              )}
                              {_vbboxes.length > 0 && (
                                <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
                                  <span style={{ fontSize:'var(--text-xs)', fontWeight:700, letterSpacing:'.08em', color:T.ink3 }}>SPATIAL ANCHORS</span>
                                  {_vbboxes.map((b, bi) => {
                                    const guide  = regionGuide(b.label);
                                    const dotCol = tierColor(guide.tier);
                                    const coach  = b.justification || guide.tip;
                                    return (
                                      <div key={bi} style={{ display:'flex', gap:8, alignItems:'flex-start',
                                        padding:'6px 10px', background:T.raised, borderRadius:'var(--r-md)',
                                        border:`1px solid ${T.line}` }}>
                                        <div style={{ width:6, height:6, borderRadius:'var(--r-round)',
                                          background:dotCol, flexShrink:0, marginTop:4 }}/>
                                        <div style={{ flex:1, minWidth:0 }}>
                                          <span style={{ fontSize:'var(--text-xs)', fontWeight:700, letterSpacing:'.06em',
                                            color:dotCol }}>{`${tierIcon(guide.tier)} ${guide.title}`.toUpperCase()}</span>
                                          {coach && (
                                            <p style={{ fontSize:'var(--text-xs)', color:T.ink2, lineHeight:1.6, margin:'2px 0 0' }}>
                                              {coach}
                                            </p>
                                          )}
                                        </div>
                                      </div>
                                    );
                                  })}
                                </div>
                              )}
                            </div>
                          );
                        })()}
                        {/* Jury critique fallback */}
                        {!rl && juryCritique && (
                          <div style={{ fontSize:'var(--text-sm)', color:T.ink2, lineHeight:1.75 }}>
                            {parseCritique(juryCritique, setCritTrigger, () => setCritTrigger(''))}
                          </div>
                        )}
                        {!rl && !juryCritique && (
                          <div style={{ display:'flex', flexDirection:'column', gap:8 }}>
                            <p style={{ fontSize:'var(--text-sm)', color:T.ink3, lineHeight:1.7 }}>No grader analysis. Generate a jury critique:</p>
                            <button
                              onClick={() => sel && handleJuryCritique(sel.path)}
                              disabled={juryLoading}
                              style={{ display:'flex', alignItems:'center', gap:6, padding:'7px 14px', borderRadius:'var(--r-md)', background:T.raised, border:`1px solid ${T.lineStrong}`, color:T.ink2, fontSize:'var(--text-sm)', fontWeight:700, cursor: juryLoading ? 'wait' : 'pointer', alignSelf:'flex-start' }}>
                              {juryLoading
                                ? <><span style={{ width:10, height:10, borderRadius:'var(--r-round)', border:`1.5px solid ${T.ink3}`, borderTopColor:'transparent', animation:'spin .8s linear infinite', display:'inline-block' }}/> Generating…</>
                                : <><Wand2 size={11}/> Write a critique</>}
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })() : (
                    <p style={{ fontSize:'var(--text-sm)', color:T.ink3, lineHeight:1.7 }}>Grade your folder to see analysis.</p>
                  )
                )}
                {infoTab === 'breakdown' && (
                  isGraded ? (
                    (() => {
                      const raw: Record<string, any> = sel?.breakdown ?? {};

                      // ── Archetype weights ─────────────────────────────────────
                      const archW = raw['_arch_w'] as Record<string,number> | undefined;
                      const ARCH_LABELS: Record<string,string> = {
                        geo:'Lines & Form', night:'Night Scene', layer:'Layered Depth', messy:'Raw Street', maxdoc:'Documentary'
                      };
                      // Archetype weight is a machine judgement, so it is cold or
                      // absent — a luminance ramp, not five competing hues. The
                      // labels sit beside the bars, so identity never depended on
                      // colour anyway.
                      const ARCH_COLORS: Record<string,string> = {
                        geo: T.ink, night: T.ink2, layer: T.ink3,
                        messy: T.ink4, maxdoc: T.lineStrong
                      };
                      let archEntries: [string,number][] = [];
                      let domArch = '';
                      if (archW) {
                        archEntries = Object.entries(archW).sort((a,b) => b[1]-a[1]);
                        domArch = archEntries[0]?.[0] ?? '';
                      }
                      const archTotal = archEntries.reduce((s,[,v]) => s+v, 0) || 1;

                      // ── Qwen one-liner critique ───────────────────────────────
                      const qwenCritique = typeof raw['_critique'] === 'string' ? (raw['_critique'] as string).trim() : '';

                      // ── Aspect bars ───────────────────────────────────────────
                      const ASPECT_KEYS = ['Technical','Composition','Lighting','Narrative','Human/Culture'] as const;
                      const SKIP_KEYS   = new Set(['aesthetic','personal','nima','_grader','_arch_w','gemma_score','vlm_bboxes','vlm_status','_critique','_tech_audit',
                        // legacy flat technical-audit keys (pre-redaction breakdowns)
                        'blur_type','highlight_clip','highlight_spread','shadow_clip','has_horizon','horizon_tilt_deg']);
                      const aspectMap: Record<string,number> = {};
                      Object.entries(raw).forEach(([k,v]) => {
                        if (!SKIP_KEYS.has(k.toLowerCase()) && !SKIP_KEYS.has(k) &&
                            typeof v === 'number' && isFinite(v as number)) {
                          aspectMap[k] = v as number;
                        }
                      });
                      // Filter v≤0.05: Qwen returns 0 for N/A aspects (e.g. Human/Culture when no person)
                      const known   = ASPECT_KEYS
                        .map(k => [k, aspectMap[k] ?? 0] as [string,number])
                        .filter(([, v]) => v > 0.05);
                      const extra   = Object.entries(aspectMap).filter(([k, v]) => !(ASPECT_KEYS as readonly string[]).includes(k) && v > 0.05);
                      const aspects = [...known, ...extra].sort((a,b) => b[1]-a[1]);
                      const best    = aspects[0]?.[0] ?? '';
                      const weakest = aspects[aspects.length - 1]?.[0] ?? '';

                      // ── Canonical dimensions (niche-agnostic) ─────────────────
                      // Resolve the five Judge's Eye rows from whatever axes this grader
                      // emitted. Qwen uses niche-specific names (e.g. "Moment", "Human",
                      // "Light Quality") that don't match the canonical keys; aspectDim
                      // maps them. Averages when several axes share a dimension; stays 0
                      // when the niche genuinely has none (row falls back to a context label).
                      const _dimAgg: Record<string,{ s:number; n:number }> = {};
                      Object.entries(aspectMap).forEach(([k, v]) => {
                        const d = aspectDim(k);
                        if (d && (v as number) > 0.05) {
                          (_dimAgg[d] ??= { s:0, n:0 });
                          _dimAgg[d].s += v as number; _dimAgg[d].n++;
                        }
                      });
                      const _dimV   = (d: string) => _dimAgg[d]?.n ? _dimAgg[d].s / _dimAgg[d].n : 0;
                      const _techV  = _dimV('tech');
                      const _lightV = _dimV('light');
                      const _hcV    = _dimV('human');
                      const _narrV  = _dimV('auth');
                      const _compV  = _dimV('comp');

                      // ── Context flags ─────────────────────────────────────────
                      const _isGeo     = domArch === 'geo'    || (archW?.geo    ?? 0) > 0.30;
                      const _isLayered = domArch === 'layer'  || (archW?.layer  ?? 0) > 0.30;
                      const _isNight   = domArch === 'night'  || (archW?.night  ?? 0) > 0.28;
                      const _isMaxDoc  = domArch === 'maxdoc' || (archW?.maxdoc ?? 0) > 0.28;
                      const _isMoody   = _narrV >= 0.38 && (_lightV || 1.0) < 0.55;
                      const _isLowKey  = _isNight || _isMoody;
                      const _isEnvShot = (_isGeo || _isMaxDoc) && _hcV < 0.05;

                      // ── Grade rationale (qualitative — no numbers) ─────────────
                      const _tier       = (sel?.grade ?? '').includes('Strong') ? 'strong'
                                        : (sel?.grade ?? '').includes('Weak')   ? 'weak'
                                        : 'mid';
                      const _gradeColor = gc(sel?.grade ?? '');
                      const _bestLabel  = aspectDim(best)    === 'human' ? 'Human presence' : (best ?? '');
                      const _weakLabel  = aspectDim(weakest) === 'human' ? 'Human presence' : (weakest ?? '');
                      const _bestPct    = best    ? Math.round((aspectMap[best]    ?? 0) * 100) : 0;
                      const _weakPct    = weakest ? Math.round((aspectMap[weakest] ?? 0) * 100) : 0;
                      const _ql = (p: number) =>
                        p >= 80 ? 'exceptional' : p >= 65 ? 'strong' : p >= 50 ? 'solid' : p >= 35 ? 'weak' : 'failing';
                      // ── Technical audit fields from backend ───────────────────
                      // Declared here (before _gradeWhy) because the reasoning IIFE
                      // below reads them. A `const` declared after the IIFE would be
                      // in the temporal dead zone when the IIFE runs, throwing a
                      // ReferenceError that unmounts the whole tree → blank screen.
                      // Technical-audit fields live under the private `_tech_audit`
                      // namespace so their raw backend names never surface as aspects.
                      // Fall back to legacy flat keys for breakdowns cached pre-redaction.
                      const _ta = (raw['_tech_audit'] as Record<string, unknown>) ?? {};
                      const _blurType        = ((_ta.blur_type        ?? raw['blur_type'])        as string)  ?? 'sharp';
                      const _hlSpread        = ((_ta.highlight_spread  ?? raw['highlight_spread']) as boolean) ?? false;
                      const _hlClip          = ((_ta.highlight_clip    ?? raw['highlight_clip'])   as number)  ?? 0;
                      const _hasHorizon      = ((_ta.has_horizon       ?? raw['has_horizon'])      as boolean) ?? false;
                      const _horizTilt       = ((_ta.horizon_tilt_deg  ?? raw['horizon_tilt_deg']) as number)  ?? 0;

                      // ── Educational grade reasoning ───────────────────────────
                      // Voice: photography mentor — explains photographic concepts,
                      // names what's working and why, names what to fix and how.
                      const _gradeWhy = (() => {
                        if (!best || aspects.length === 0) return '';

                        // Technical audit context for reasoning
                        const _shakeBlur = _blurType === 'shake' || _blurType === 'severe';
                        const _bokehBlur = _blurType === 'bokeh';
                        const _panBlur   = _blurType === 'panning';
                        const _isTilted = _hasHorizon && _horizTilt > 3;

                        if (_tier === 'strong') {
                          if (best === 'Narrative' && _bestPct >= 70)
                            return `Decisive moment caught — gesture, light, and composition align in the same frame. That's the hardest thing to do consistently in street photography.`;
                          if (best === 'Composition' && _isGeo)
                            return `The geometric structure carries the frame. Strong compositional intentionality — leading lines, form, and light are working together.`;
                          if (best === 'Lighting' && _isLowKey)
                            return `Chiaroscuro — the darkness is doing compositional work, directing attention to the lit subject. Shadow is a tool here, not a failure.`;
                          if (best === 'Composition' && _isLayered)
                            return `Foreground-background layering creates depth. The compressed perspective pulls the viewer through the frame.`;
                          if (_bestPct >= 75)
                            return `${_bestLabel} is at portfolio level — that's what defines this frame. No critical failures pulling it down.`;
                          return `No critical failures, and ${_bestLabel} is doing portfolio-level work. That's the formula for a consistent keeper.`;
                        }

                        if (_tier === 'weak') {
                          if (_shakeBlur)
                            return `Camera shake is destroying the frame — unrecoverable. At street focal lengths, 1/250s is a safe minimum shutter. If light is low, raise ISO before slowing down.`;
                          if (_hlSpread)
                            return `Blown highlights are pulling the viewer's eye to empty white areas. Expose for the bright areas and recover the shadows in post — you can't recover clipped highlights.`;
                          if (aspectDim(weakest) === 'auth' && _weakPct < 35 && _hcV > 0.2)
                            return `A person walking past isn't a decisive moment. Street photography is about the split second when gesture, light, and geometry say something together. Here, nothing has happened yet.`;
                          if (_blurType === 'severe' || _weakPct < 32)
                            return `${_weakLabel} is failing — the frame can't recover from it. ${_bestPct >= 55 ? `${_bestLabel} shows real intent, but one catastrophic failure outweighs everything else.` : 'Rebuild from the technical floor first.'}`;
                          return `${_weakLabel} is the floor — too weak to pull the grade up. Identify and solve that one problem and revisit this frame's potential.`;
                        }

                        // Mid — explain the ceiling and how to break through it
                        if (_shakeBlur)
                          return `Camera shake is the ceiling — it caps the technical score and drags down everything else. Fix the shutter speed first; the other dimensions have potential.`;
                        if (aspectDim(weakest) === 'auth' && _weakPct < 45 && _hcV > 0.2)
                          return `The light and frame are ready but the decisive moment hasn't arrived. Pre-focus and wait — the geometry is already there for a strong frame when something happens.`;
                        if (aspectDim(weakest) === 'tech' && _weakPct < 50)
                          return `The intent is here but the technical floor is the ceiling. At this focal length, a faster shutter or steadier hold would lock in the sharpness that's missing.`;
                        if (aspectDim(weakest) === 'human' && _weakPct < 45 && !_isEnvShot)
                          return `Strong environmental frame — the light and geometry are working — but it needs a human anchor. Even a peripheral figure adds scale and completes the narrative.`;
                        if (_hlSpread)
                          return `Blown highlights are competing with the subject for attention. In harsh light, expose for the bright areas — protected highlights read cleaner than blown ones.`;
                        if (_isTilted && _isGeo)
                          return `The horizon tilt reads as accidental rather than intentional. Commit to a perfectly level frame or a deliberate extreme Dutch angle — the in-between is ambiguous.`;
                        if (_bestPct >= 65 && _weakPct < 50)
                          return `${_bestLabel} is at a strong level. ${_weakLabel} is the ceiling — bringing that up would push this to portfolio-worthy.`;
                        if (_bestPct >= 60)
                          return `Good instincts across the board — no single dimension is carrying it. Identify one element to lead and build the frame around it deliberately.`;
                        return `Technically acceptable but no dimension reaches portfolio level yet. Study what's strongest here and push it further in the next frame.`;
                      })();

                      // ── Evidence Checklist ─────────────────────────────────────
                      type _CS = 'good' | 'ok' | 'bad' | 'neutral';
                      // _techV/_lightV/_hcV/_narrV/_compV resolved above via aspectDim.
                      const _vlmSt  = (raw['vlm_status'] as string) ?? '';

                      // Technical audit fields (_blurType/_hlSpread/_hlClip/_hasHorizon/
                      // _horizTilt) are declared above, before _gradeWhy.

                      // Map the weakest VLM aspect to its checklist row via its dimension,
                      // so niche-specific axis names ("Moment", "Light Quality"…) resolve too.
                      const _DIM_TO_CHECK: Record<string,string> = {
                        tech:'FOCUS', light:'EXPOSURE', human:'SUBJECT', auth:'MOMENT', comp:'GEOMETRY',
                      };
                      const _limitingCheck = _DIM_TO_CHECK[aspectDim(weakest)] ?? '';

                      const _checks: Array<{ key:string; label:string; value:string; state:_CS; isLimit:boolean; note:string }> = [];

                      // FOCUS — blur_type from technical audit takes priority over TOPIQ range labels
                      {
                        let s: _CS; let v: string; let note: string;
                        if (_blurType === 'severe' || _vlmSt === 'CRITICAL_BLUR') {
                          s = 'bad'; v = 'Severe Blur';
                          note = 'Unrecoverable. Check that autofocus locked before shooting, or that the shutter was fast enough.';
                        } else if (_blurType === 'shake') {
                          s = 'bad'; v = 'Camera Shake';
                          note = 'Minimum safe shutter is 1/focal length. If light is low, raise ISO before slowing the shutter.';
                        } else if (_blurType === 'bokeh') {
                          s = 'good'; v = 'Subject Sharp / Bokeh';
                          note = 'Intentional depth of field — subject isolated against a defocused background. This is a technique, not a flaw.';
                        } else if (_blurType === 'panning') {
                          s = 'ok'; v = 'Intentional Motion Blur';
                          note = 'Panning blur conveys speed and energy. The subject stays sharp while the background streaks — a deliberate choice.';
                        } else {
                          s = _techV >= 0.72 ? 'good' : _techV >= 0.50 ? 'ok' : 'bad';
                          v = _techV >= 0.72 ? 'Tack-Sharp' : _techV >= 0.50 ? 'Acceptable' : 'Soft';
                          note = _techV >= 0.72 ? 'Focus is locked — technical floor is solid.'
                            : _techV >= 0.50 ? 'Usable sharpness. A slightly faster shutter or smaller aperture would sharpen this.'
                            : 'The focus plane may have shifted. Zone focus or a smaller aperture increases your keeper rate.';
                        }
                        _checks.push({ key:'FOCUS', label:'Focus', value:v, state:s, isLimit:_limitingCheck === 'FOCUS' && _tier !== 'strong', note });
                      }
                      // EXPOSURE — highlight_spread overrides lighting score; night/moody exempted
                      {
                        let s: _CS; let v: string; let note: string;
                        if (_lightV < 0.05) {
                          s = 'neutral'; v = 'Not Scored';
                          note = 'This style is judged on its core dimensions — light reads as part of the overall craft rather than a separate score.';
                        } else if (_hlSpread && !_isLowKey) {
                          s = 'bad'; v = `Blown Highlights (${Math.round(_hlClip * 100)}% clipped)`;
                          note = 'Clipped highlights can\'t be recovered in post. Expose for the bright areas and lift the shadows instead.';
                        } else if (_isLowKey) {
                          s = _lightV >= 0.65 ? 'good' : 'ok';
                          v = _lightV >= 0.65 ? 'Protected'
                            : _lightV >= 0.55 ? 'Atmospheric'
                            : _lightV >= 0.40 ? 'Low-Key / Intentional'
                            : 'Deep Shadow';
                          note = _lightV >= 0.65
                            ? 'Shadow detail is preserved — the dark areas are intentional, not a failure.'
                            : 'Low-key exposure is the artistic choice here. The darkness is directing attention.';
                        } else {
                          s = _lightV >= 0.65 ? 'good' : _lightV >= 0.50 ? 'ok' : _lightV >= 0.35 ? 'ok' : 'bad';
                          v = _lightV >= 0.65 ? 'Protected'
                            : _lightV >= 0.50 ? 'Acceptable'
                            : _lightV >= 0.35 ? 'Flat / Weak'
                            : 'Blown / Dead-flat';
                          note = _lightV >= 0.65 ? 'Exposure is controlled — highlights and shadows both readable.'
                            : _lightV >= 0.50 ? 'Workable light. Look for directional sources to add dimension.'
                            : _lightV >= 0.35 ? 'Flat light is workable — use it for even portraits or lean into the mood deliberately.'
                            : 'Light is too flat or blown. Reposition relative to the light source.';
                        }
                        _checks.push({ key:'EXPOSURE', label:'Exposure', value:v, state:s, isLimit:_limitingCheck === 'EXPOSURE' && _tier !== 'strong', note });
                      }
                      // SUBJECT — environmental/geo shots don't need a subject
                      if (_hcV > 0.05) {
                        const s: _CS = _hcV >= 0.65 ? 'good' : _hcV >= 0.50 ? 'ok' : 'bad';
                        const v = _hcV >= 0.65 ? 'Active Presence' : _hcV >= 0.50 ? 'Incidental Figure' : 'Background Only';
                        const note = _hcV >= 0.65 ? 'The human element is doing narrative work — gesture or expression is present.'
                          : _hcV >= 0.50 ? 'Subject is present but peripheral. Closer proximity or a stronger gesture would anchor the frame.'
                          : 'Subject is lost in the scene. Move closer or wait for the subject to separate from the background.';
                        _checks.push({ key:'SUBJECT', label:'Subject', value:v, state:s, isLimit:_limitingCheck === 'SUBJECT' && _tier !== 'strong', note });
                      } else if (_isEnvShot) {
                        _checks.push({ key:'SUBJECT', label:'Subject', value:'Not Required', state:'neutral', isLimit:false,
                          note:'Environmental frame — geometry and light carry the image without a human anchor.' });
                      } else {
                        _checks.push({ key:'SUBJECT', label:'Subject', value:'Empty Scene', state:'neutral', isLimit:_limitingCheck === 'SUBJECT' && _tier !== 'strong',
                          note:'No subject detected. Even a peripheral figure adds scale and narrative to an empty scene.' });
                      }
                      // MOMENT — always shown; N/A for pure architectural (Qwen returns 0)
                      {
                        let s: _CS; let v: string; let note: string;
                        if (_narrV < 0.05) {
                          s = 'neutral'; v = _isEnvShot ? 'Not Applicable' : 'Not Scored';
                          note = _isEnvShot
                            ? 'Architectural and environmental frames are judged on geometry and light, not narrative moment.'
                            : 'This style is judged on its core dimensions — narrative moment isn\'t scored separately here.';
                        } else {
                          s = _narrV >= 0.65 ? 'good' : _narrV >= 0.50 ? 'ok' : 'bad';
                          v = _narrV >= 0.65 ? 'Decisive Moment'
                            : _narrV >= 0.50 ? 'Something Happening'
                            : 'Static / No Tension';
                          note = _narrV >= 0.65
                            ? 'Peak gesture or expression caught — this is the hardest thing to do consistently in street photography.'
                            : _narrV >= 0.50
                            ? 'Action is present. Pushing closer to the peak of the gesture would strengthen the narrative.'
                            : 'Nothing decisive has happened yet. Anticipate — pre-focus and wait for the moment when body language, light, and geometry converge.';
                        }
                        _checks.push({ key:'MOMENT', label:'Moment', value:v, state:s, isLimit:_limitingCheck === 'MOMENT' && _tier !== 'strong', note });
                      }
                      // GEOMETRY — horizon tilt takes priority when a real horizon is detected
                      {
                        let s: _CS; let v: string; let note: string;
                        if (_hasHorizon && _horizTilt > 10) {
                          s = 'bad'; v = `Excessive Tilt (${_horizTilt}°)`;
                          note = 'Beyond intentional Dutch angle range. Straighten the horizon or commit to an extreme tilt — the in-between reads as accidental.';
                        } else if (_hasHorizon && _isGeo && _horizTilt > 3) {
                          s = 'bad'; v = `Tilted Horizon (${_horizTilt}°)`;
                          note = 'In geometric and architectural work, a level horizon is part of the craft. A few degrees of correction would resolve this.';
                        } else if (_hasHorizon && _horizTilt > 3 && !_isGeo) {
                          s = 'ok'; v = `Dynamic Tilt (${_horizTilt}°)`;
                          note = 'Winogrand-style tilt adds energy and instability. Works in street — just make sure it reads as intentional.';
                        } else {
                          const _geoStrict = _isGeo && _compV < 0.40;
                          s = _compV >= 0.65 ? 'good' : _compV >= 0.50 ? 'ok' : _geoStrict ? 'bad' : 'ok';
                          v = _isGeo && _compV >= 0.65 ? 'Leading Lines / Frame-within-Frame'
                            : _isLayered && _compV >= 0.65 ? 'Foreground + Background Depth'
                            : _compV >= 0.65 ? 'Clean Hierarchy'
                            : _compV >= 0.50 ? 'Serviceable Frame'
                            : 'Loose Frame';
                          note = _compV >= 0.65
                            ? 'Compositional intentionality is clear — the frame directs the viewer\'s eye deliberately.'
                            : _compV >= 0.50
                            ? 'Structure is there. A tighter crop or a more deliberate entry point would sharpen the composition.'
                            : 'The frame is loose — the viewer\'s eye doesn\'t have a clear path. Find a foreground element or a stronger subject relationship.';
                        }
                        _checks.push({ key:'GEOMETRY', label:'Geometry', value:v, state:s, isLimit:_limitingCheck === 'GEOMETRY' && _tier !== 'strong', note });
                      }

                      const _csCol = (s: _CS) =>
                        s === 'good' ? T.gradeStrong : s === 'ok' ? T.ink2 : s === 'bad' ? T.gradeWeak : T.ink3;

                      // ── Telemetry Tags ─────────────────────────────────────────
                      const _ARCH_TAG: Record<string,{ icon:string; detail:string }> = {
                        geo:    { icon:'', detail:'Leading lines and symmetry' },
                        layer:  { icon:'', detail:'Layered depth, foreground against background' },
                        night:  { icon:'', detail:'Night and low light, deep shadow' },
                        messy:  { icon:'', detail:'Raw street energy' },
                        maxdoc: { icon:'', detail:'Dense documentary coverage' },
                      };
                      const _teleTags: Array<{ key:string; label:string; icon:string; detail:string; dominant:boolean }> = [];
                      // Build per-image detail strings from actual aspect scores + archetype weight
                      archEntries.filter(([, v]) => v > 0.15).forEach(([k, kv], i) => {
                        const t = _ARCH_TAG[k];
                        if (!t) return;
                        const w = Math.round((kv as number) * 100);
                        let detail = t.detail;
                        if (k === 'geo') {
                          detail = _compV >= 0.65
                            ? `Strong geometric structure — ${w}% geometric signal`
                            : `Geometric intent present — composition needs tightening`;
                        } else if (k === 'layer') {
                          detail = _hcV >= 0.65
                            ? `Subject isolated in depth — foreground/background contrast working`
                            : _hcV >= 0.50
                            ? `Layering present — subject needs stronger separation`
                            : `Layered scene — no clear subject anchor`;
                        } else if (k === 'night') {
                          detail = _lightV >= 0.55
                            ? `Low-key lighting controlled — shadow detail preserved`
                            : _lightV >= 0.40
                            ? `Night / available light — exposure borderline`
                            : `Dark scene — underexposure risk`;
                        } else if (k === 'messy') {
                          detail = _narrV >= 0.65
                            ? `Chaotic energy — decisive moment caught in the disorder`
                            : `Raw street chaos — no clear moment anchors it`;
                        } else if (k === 'maxdoc') {
                          detail = _compV >= 0.65
                            ? `Dense documentary — layered scene with readable hierarchy`
                            : `Documentary density — frame needs stronger visual entry point`;
                        }
                        _teleTags.push({ key:k, label:ARCH_LABELS[k] ?? k, icon:t.icon, detail, dominant:i === 0 });
                      });
                      if (_narrV >= 0.55) _teleTags.push({
                        key:'timing', label:'Timing', icon:'⏳',
                        detail: _narrV >= 0.75
                          ? `Peak decisive moment — gesture or tension locked at exactly the right frame`
                          : `Active moment caught — scene energy is readable`,
                        dominant: _teleTags.length === 0,
                      });

                      // ── Burst Context ──────────────────────────────────────────
                      const _simFlag   = (sel?.sim_flag as string) ?? '';
                      const _isBurst   = (sel?.cluster_id ?? -1) >= 0 && _simFlag.length > 0;
                      const _isPrimary = _simFlag.startsWith('★');
                      const _burstCntM = _simFlag.match(/Best of (\d+)/);
                      const _burstCnt  = _burstCntM ? parseInt(_burstCntM[1]) : 0;
                      const _altM      = _simFlag.match(/Duplicate — (.+?) is better/);
                      const _altName   = _altM ? _altM[1] : '';

                      return (
                        <div className="animate-fade-in flex flex-col gap-4">

                          {/* ── Burst Context ─────────────────────────────────── */}
                          {_isBurst && (
                            <div className="rounded-sm border border-line bg-surface p-3">
                              <p className="t-label mb-1">Burst selection</p>
                              <div className="mb-1 flex items-baseline gap-2">
                                <span className={cn('text-sm font-semibold',
                                                    _isPrimary ? 'text-ink' : 'text-ink-2')}>
                                  {_isPrimary ? 'Primary pick' : 'Alternate'}
                                </span>
                                {_isPrimary && _burstCnt > 0 && (
                                  <span className="text-xs text-ink-3">
                                    of <span className="t-num">{_burstCnt}</span> similar frames
                                  </span>
                                )}
                              </div>
                              <p className="text-xs text-ink-2">
                                {_isPrimary
                                  ? 'Highest-scoring frame in this burst. The others scored lower overall.'
                                  : `${_altName || 'Another frame'} scored higher and is the primary pick. Compare before rejecting.`}
                              </p>
                            </div>
                          )}

                          {/* ── The verdict ───────────────────────────────────────
                           * Score set as a typographic object rather than a badge
                           * on a tinted panel: `.612` large, in mono, leading zero
                           * dropped the way f/1.4 and 1/250 drop theirs. The grade
                           * name sits under it as a quiet label, and the only
                           * colour is the thin rule — the same rule used in the
                           * contact sheet, so the two views speak one language. */}
                          <div className="flex flex-col gap-2">
                            <div className="flex items-baseline gap-3">
                              <span className="t-num text-xl leading-none text-ink">
                                {formatScore(sel?.score)}
                              </span>
                              <span className="t-label">{gradeLabel(sel?.grade)}</span>
                            </div>
                            <span aria-hidden className="block w-full"
                                  style={{ height: 'var(--rule)',
                                           background: gradeRule(sel?.grade) ?? T.line }}/>
                            {_gradeWhy && (
                              <p className="text-sm text-ink-2">{_gradeWhy}</p>
                            )}
                          </div>

                          {/* ── Model one-liner ───────────────────────────────── */}
                          {qwenCritique && (
                            <p className="border-l-2 border-line-strong pl-3 text-sm text-ink-3">
                              {qwenCritique}
                            </p>
                          )}

                          {/* ── Evidence Checklist ────────────────────────────── */}
                          {_checks.length > 0 && (
                            <div>
                              <div style={{ display:'flex', alignItems:'baseline', gap:8, marginBottom:8 }}>
                                <span style={{ fontSize:'var(--text-xs)', fontWeight:700, letterSpacing:'.09em',
                                  textTransform:'uppercase', color:T.ink3 }}>Judge's Eye</span>
                                <span style={{ fontSize:'var(--text-xs)', color:T.ink3, opacity:.6 }}>— what's working and what to fix</span>
                              </div>
                              <div style={{ display:'flex', flexDirection:'column',
                                borderRadius:'var(--r-md)', overflow:'hidden', border:`1px solid ${T.line}` }}>
                                {_checks.map(({ key, label, value, state, isLimit, note }, ci) => {
                                  const col = _csCol(state);
                                  const showNote = !!note && (state === 'bad' || isLimit || _tier !== 'strong');
                                  return (
                                    <div key={key} style={{
                                      padding:'8px 11px',
                                      // A limit row is emphasised by luminance, not
                                      // by a red or amber wash behind the text.
                                      background: isLimit ? T.raisedHover
                                        : ci % 2 === 0 ? T.raised : T.ground,
                                      borderBottom: ci < _checks.length - 1 ? `1px solid ${T.line}` : 'none' }}>
                                      <div style={{ display:'flex', alignItems:'center', gap:10 }}>
                                        <div style={{ width:6, height:6, borderRadius:'var(--r-round)',
                                          background:col, flexShrink:0 }}/>
                                        <span style={{ fontSize:'var(--text-xs)', fontWeight:700, letterSpacing:'.07em',
                                          color:T.ink3, minWidth:62, textTransform:'uppercase' }}>{label}</span>
                                        <span style={{ fontSize:'var(--text-xs)', fontWeight:600, color:col }}>{value}</span>
                                        {isLimit && (
                                          <span style={{ marginLeft:'auto', fontSize:'var(--text-xs)', fontWeight:800,
                                            letterSpacing:'.08em', color: _tier === 'weak' ? T.gradeWeak : T.ink2,
                                            textTransform:'uppercase' }}>
                                            {_tier === 'weak' ? '↑ WHAT FAILED' : '↑ WHAT TO FIX'}
                                          </span>
                                        )}
                                      </div>
                                      {showNote && (
                                        <p style={{ fontSize:'var(--text-xs)', color:T.ink3, lineHeight:1.55,
                                          margin:'5px 0 0 16px', fontStyle:'italic' }}>
                                          {note}
                                        </p>
                                      )}
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          )}

                          {/* Visual language. The emoji that used to lead each row
                              (📐 👥 🌙 ⚡ 📰) are gone — emoji as a category marker
                              is decoration, and at 13px they read as clip-art next
                              to the photograph they are describing. The dominant
                              row is marked by a rule and brighter ink instead. */}
                          {_teleTags.length > 0 && (
                            <div className="flex flex-col gap-2">
                              <div>
                                <p className="t-label">Visual language</p>
                                <p className="mt-px text-xs text-ink-3">
                                  The photographic tradition this frame is working in.
                                </p>
                              </div>
                              <div className="flex flex-col gap-1">
                                {_teleTags.map(({ key, label, detail, dominant }) => (
                                  <div key={key}
                                    className={cn('flex items-baseline gap-2 rounded-sm border px-3 py-2',
                                                  dominant
                                                    ? 'border-line-strong bg-raised'
                                                    : 'border-line bg-surface')}>
                                    <span aria-hidden className="w-2 shrink-0 self-center"
                                          style={{ height: 'var(--rule)',
                                                   background: dominant ? T.ink2 : T.line }}/>
                                    <span className={cn('t-label shrink-0',
                                                        dominant ? '!text-ink' : '!text-ink-2')}>
                                      {label}
                                    </span>
                                    <span className="min-w-0 flex-1 text-xs text-ink-3">{detail}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                        </div>
                      );
                    })()
                  ) : (
                    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:8, padding:'20px 0' }}>
                      <Layers size={24} strokeWidth={1} style={{ color:T.ink3 }}/>
                      <p style={{ fontSize:'var(--text-sm)', color:T.ink3, textAlign:'center', lineHeight:1.6 }}>Grade your folder to see breakdown.</p>
                    </div>
                  )
                )}
              </div>

            </div>}

            </>)}
          </div>

          {/* ── Filmstrip (loupe mode only) ─────────────────────── */}
          {loupeMode === 'loupe' && photos.length > 0 && (
          <div style={{ flexShrink:0, background:T.surface, borderTop:`1px solid ${T.line}`, display:'flex', flexDirection:'column' }}>
            <div style={{ height:20, flexShrink:0, display:'flex', alignItems:'center', justifyContent:'space-between', padding:'0 12px', borderBottom:`1px solid ${T.line}` }}>
              <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                <span style={{ fontSize:'var(--text-xs)', color:T.ink3, fontWeight:600, letterSpacing:'.08em', textTransform:'uppercase' }}>Library</span>
                {/* Tweaks toggle */}
                <button title="Filmstrip settings" onClick={() => setShowTweaks(v => !v)}
                  style={{ display:'flex', alignItems:'center', justifyContent:'center', width:18, height:16, cursor:'pointer', background:showTweaks ? T.raisedHover : 'transparent', color:showTweaks ? T.ink : T.ink3, border:'none', borderRadius:'var(--r-sm)', transition:'all .25s cubic-bezier(.2,0,0,1)' }}>
                  <SlidersHorizontal size={9}/>
                </button>
              </div>
              <span style={{ fontSize:'var(--text-xs)', color:T.ink3, fontVariantNumeric:'tabular-nums', display:'flex', alignItems:'center', gap:5 }}>
                {isGrading && <span style={{ display:'inline-block', width:5, height:5, border:`1.5px solid ${T.ink3}`, borderTopColor:'transparent', borderRadius:'var(--r-round)', animation:'spin .8s linear infinite' }}/>}
                {isDone
                  ? <><span style={{ color:T.gradeStrong }}>{picks} picks</span>{'  ·  '}<span style={{ color:T.gradeWeak }}>{rejects} rejects</span>{'  ·  '}{photos.length} total</>
                  : `${photos.length} photos`}
              </span>
            </div>
            {/* Tweaks panel */}
            {showTweaks && (
              <div style={{ flexShrink:0, display:'flex', alignItems:'center', gap:16, padding:'6px 12px', borderBottom:`1px solid ${T.line}`, background:T.raised }}>
                <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                  <span style={{ fontSize:'var(--text-xs)', color:T.ink3, whiteSpace:'nowrap' }}>Thumb size</span>
                  <input type="range" min={60} max={130} step={4} value={filmThumbH}
                    onChange={e => setFilmThumbH(Number(e.target.value))}
                    style={{ width:80, accentColor:T.ink, cursor:'pointer' }}/>
                  <span style={{ fontSize:'var(--text-xs)', color:T.ink2, fontVariantNumeric:'tabular-nums', minWidth:22 }}>{filmThumbH}</span>
                </div>
                <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                  <span style={{ fontSize:'var(--text-xs)', color:T.ink3 }}>Filenames</span>
                  <button onClick={() => setShowFilename(v => !v)}
                    style={{ position:'relative', width:28, height:16, borderRadius:'var(--r-md)', border:'none', cursor:'pointer', padding:0, background:showFilename ? T.ink : T.lineStrong, transition:'background .25s ease' }}>
                    <span style={{ position:'absolute', top:2, left:showFilename ? 13 : 2, width:12, height:12, borderRadius:'var(--r-round)', background:T.ink, transition:'left .22s cubic-bezier(.2,0,0,1)', boxShadow:`0 1px 2px ${T.well}` }}/>
                  </button>
                </div>
              </div>
            )}
            <div ref={filmRef} style={{ height: filmThumbH + (showFilename ? 18 : 0) + 12, overflowX:'auto', overflowY:'hidden', display:'flex', alignItems:'center', padding:'0 6px', gap:4 }}>
              {filteredPhotos.map(p => (
                <FilmThumb key={p.id} p={p} isSel={p.id === selId} onSelect={setSelId} isUsed={allUsedPaths.has(p.path)} isSelected={selectedIds.has(p.id)} h={filmThumbH} showFn={showFilename}/>
              ))}
            </div>
          </div>
          )}
        </div>

      ) : mainTab === 'duplicates' ? (
        /* ── Duplicates grid view ──────────────────────────────── */
        (() => {
          const byCluster: Record<number, any[]> = {};
          for (const p of photos) {
            if (p.cluster_id < 0) continue;
            (byCluster[p.cluster_id] ??= []).push(p);
          }
          const groups = Object.values(byCluster)
            .map(g => {
              const best = g.find(p => (p.sim_flag||'').includes('Best')) ?? g[0];
              const rest = g.filter(p => p !== best).sort((a,b) => b.score - a.score);
              return { best, rest, all: [best, ...rest] };
            })
            .sort((a, b) => b.all.length - a.all.length);
          const totalDups = groups.reduce((s, g) => s + g.rest.length, 0);

          return (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden bg-ground">
              <div className="flex h-8 shrink-0 items-center gap-2 border-b border-line bg-surface px-4">
                <span className="text-md font-semibold text-ink">Similar shots</span>
                <span className="text-xs text-ink-3">
                  <span className="t-num">{groups.length}</span> group{groups.length!==1?'s':''}
                  {' · '}<span className="t-num">{totalDups}</span> alternates
                </span>
                <div className="ml-auto">
                  <Button size="sm" onClick={() => setExportModal(true)} icon={<Download size={11}/>}>
                    Export
                  </Button>
                </div>
              </div>

              <div className="min-h-0 flex-1 overflow-y-auto p-3">
                {groups.map((g, gi) => {
                  const bestRule = gradeRule(g.best.grade);
                  return (
                    <div key={gi} className={cn(gi < groups.length - 1 && 'mb-6')}>

                      {/* Group label. The rule colour matches the winner's grade,
                          so a burst whose best frame is only Mid reads as such
                          before you look at a single thumbnail. */}
                      <div className="mb-2 flex items-center gap-2">
                        <span aria-hidden className="w-4 shrink-0"
                              style={{ height: 'var(--rule)', background: bestRule ?? T.line }}/>
                        <span className="t-label !text-ink-2">
                          <span className="t-num">{g.all.length}</span> similar
                        </span>
                        <span className="text-xs text-ink-3">
                          best <span className="t-num text-ink">{formatScore(g.best.score)}</span>
                        </span>
                        <div className="h-px flex-1 bg-line"/>
                        <Button size="sm" variant="quiet"
                          onClick={() => { setMainTab('gallery'); setSelId(g.best.id); setLoupeMode('loupe'); }}>
                          Open best
                        </Button>
                      </div>

                      {/* Frames keep native aspect here too. Comparing near-identical
                          frames is the ENTIRE job of this screen, so cropping them all
                          to the same rectangle destroyed the one difference — framing —
                          you are here to judge. */}
                      <div className="flex flex-wrap gap-1">
                        {g.all.map((p: any, pi: number) => {
                          const isBest = pi === 0;
                          const delta  = isBest ? null : p.score - g.best.score;
                          const fname  = (p.path.split(/[\\/]/).pop() ?? '').replace(/\.[^.]+$/, '');
                          return (
                            <button key={p.id}
                              onClick={() => { setMainTab('gallery'); setSelId(p.id); setLoupeMode('loupe'); }}
                              className={cn(
                                'flex cursor-pointer flex-col border-0 bg-surface p-0',
                                'rounded-sm outline outline-2 outline-offset-1',
                                'transition-[outline-color] duration-fast ease',
                                isBest ? 'outline-ink' : 'outline-transparent hover:outline-line-strong',
                              )}>

                              <span className="relative block overflow-hidden bg-well" style={{ height: 116 }}>
                                <img src={thumbUrl(p.path)} alt="" loading="lazy"
                                  className={cn('block h-full w-auto max-w-none',
                                                !isBest && 'opacity-reject')}/>
                              </span>

                              {/* Best is marked by a rule, matching the contact sheet.
                                  No scrim, no floating badges — nothing covers a frame
                                  you are trying to compare. */}
                              <span aria-hidden className="block w-full"
                                    style={{ height: 'var(--rule)',
                                             background: isBest ? (bestRule ?? T.ink3) : 'transparent' }}/>

                              <span className="flex items-center gap-1 px-1 py-px">
                                <span className={cn('t-num flex-1 truncate text-left text-xs',
                                                    isBest ? 'text-ink-2' : 'text-ink-4')}>
                                  {fname}
                                </span>
                                {delta !== null && (
                                  <span className="t-num shrink-0 text-xs text-ink-4"
                                        title="Score difference from the best frame">
                                    {delta > 0 ? '+' : '−'}{formatScore(Math.abs(delta))}
                                  </span>
                                )}
                                <span className={cn('t-num shrink-0 text-xs',
                                                    isBest ? 'text-ink' : 'text-ink-3')}>
                                  {formatScore(p.score)}
                                </span>
                              </span>

                            </button>
                          );
                        })}
                      </div>

                    </div>
                  );
                })}

                {groups.length === 0 && (
                  <div className="flex flex-col items-center justify-center gap-3 pt-12 text-ink-3">
                    <ImageOff size={28} strokeWidth={1}/>
                    <p className="text-sm">No similar shots found.</p>
                    <p className="max-w-[36ch] text-center text-xs text-ink-4">
                      Bursts and near-duplicates appear here once a folder has been graded.
                    </p>
                  </div>
                )}
              </div>
            </div>
          );
        })()

      ) : mainTab === 'creative' ? (
        /* ── Creative Direction view ───────────────────────────── */
        (() => {
          const SLOT_COLORS: Record<string,string> = {
            // Sequence roles are assigned by the machine, so they read as a
            // luminance ramp rather than five hues competing with the frames.
            Opener:   T.ink,
            Subject:  T.ink,
            Contrast: T.ink2,
            Detail:   T.ink3,
            Closer:   T.ink2,
          };
          const slotColor = (s: string) => SLOT_COLORS[s] ?? SLOT_COLORS[(s||'').charAt(0).toUpperCase()+(s||'').slice(1)] ?? T.ink3;
          const ROLE_ORDER = ['Opener','Subject','Contrast','Detail','Closer','opener','subject','contrast','detail','closer'];
          const sortedPhotos = [...photos].sort((a,b) => {
            const r = (p:any) => gradeLabel(p.grade)==='Strong'?0:gradeLabel(p.grade)==='Mid'?1:2;
            return r(a)-r(b) || b.score-a.score;
          });
          const successResults = [...creativeResults.filter((r:any)=>r.success)]
            .sort((a:any,b:any) => {
              const ap = a.params?.seq_pos; const bp = b.params?.seq_pos;
              if (ap!=null && bp!=null) return ap-bp;
              const ai = ROLE_ORDER.indexOf(a.slot??a.params?.role??'');
              const bi = ROLE_ORDER.indexOf(b.slot??b.params?.role??'');
              return (ai<0?99:ai)-(bi<0?99:bi);
            });
          const hasResults = successResults.length > 0;
          const canGenerate = !creativeLoading && photos.length > 0 && engineHealth.status === 'online';

          return (
          <div className="flex flex-1 overflow-hidden bg-ground">

            {/* ── Left config panel ───────────────────────────────── */}
            <div className="flex w-panel shrink-0 flex-col overflow-hidden border-r border-line bg-surface">

              <div className="shrink-0 border-b border-line px-4 py-3">
                <div className="mb-1 flex items-center gap-2">
                  <Wand2 size={14} className="text-ink-3"/>
                  <span className="text-md font-semibold text-ink">Creative director</span>
                </div>
                <p className="text-xs text-ink-3">
                  Builds a story arc from five visually distinct frames.
                </p>
              </div>

              <div className="flex flex-1 flex-col gap-6 overflow-y-auto px-4 py-4">

                <Field label="Mood / story brief">
                  <TextArea
                    value={creativePrompt}
                    onChange={e=>setCreativePrompt(e.target.value)}
                    placeholder={`Describe the mood…\ne.g. "rainy evening, neon reflections"\nor "empty streets at dawn"`}
                    rows={4}
                  />
                </Field>

                <Field
                  label="Reference PDFs"
                  hint="Phrases are extracted from each book and blended into the grading anchor at 30%. The books themselves are never needed again afterwards."
                  action={ragPdfs.length > 0 && (
                    <Button size="sm" variant="quiet" onClick={handleRagClear}>Clear all</Button>
                  )}
                >
                  {ragPdfs.length > 0 && (
                    <div className="flex flex-col gap-1">
                      {ragPdfs.map(p => (
                        <div key={p.name} className="flex items-center gap-2 rounded-sm border border-line-strong bg-raised px-2 py-1">
                          <span className="flex-1 truncate text-xs text-ink-2">{p.name}</span>
                          <span className="t-num shrink-0 text-xs text-ink-3">{p.phrases}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <label className={cn(
                    'flex items-center gap-2 rounded-sm border border-dashed px-3 py-2 text-sm',
                    'transition-colors duration-fast ease',
                    ragUploading
                      ? 'cursor-wait border-ink-4 text-ink-2'
                      : 'cursor-pointer border-line-strong text-ink-3 hover:border-ink-4 hover:text-ink-2',
                  )}>
                    <Upload size={13} strokeWidth={1.5}/>
                    <span>{ragUploading ? 'Reading the book…' : 'Add a reference PDF'}</span>
                    <input type="file" accept="application/pdf" className="hidden" disabled={ragUploading}
                      onChange={e => { const f = e.target.files?.[0]; if (f) handleRagUpload(f); e.target.value = ''; }}
                    />
                  </label>
                </Field>

                <Field label="Reference peg" hint="Optional. Overrides the anchor pool.">
                  {pegFile ? (
                    <div className={cn('flex items-center gap-2 rounded-sm border bg-raised px-2 py-2',
                                       pegHash ? 'border-ink-4' : 'border-line-strong')}>
                      <span className="flex-1 truncate text-xs text-ink-2">{pegFile.name}</span>
                      {pegLoading && <span className="t-label shrink-0">Reading</span>}
                      <button onClick={() => { setPegFile(null); setPegHash(null); }}
                        aria-label="Remove reference peg"
                        className="shrink-0 cursor-pointer border-0 bg-transparent p-0 text-ink-3 transition-colors duration-fast ease hover:text-ink">
                        <X size={12}/>
                      </button>
                    </div>
                  ) : (
                    <label className="flex cursor-pointer items-center gap-2 rounded-sm border border-dashed border-line-strong px-3 py-2 text-sm text-ink-3 transition-colors duration-fast ease hover:border-ink-4 hover:text-ink-2">
                      <Upload size={13} strokeWidth={1.5}/>
                      <span>Upload a reference image</span>
                      <input type="file" accept="image/*" className="hidden"
                        onChange={e => { const f = e.target.files?.[0]; if (f) handlePegUpload(f); e.target.value = ''; }}
                      />
                    </label>
                  )}
                </Field>

                {/* Sequence length */}
                <div>
                  <label className="t-label mb-2 block">
                    Sequence length
                  </label>
                  <select value={creativeCount} onChange={e => setCreativeCount(Number(e.target.value))}
                    style={{ width:'100%', height:36, borderRadius:'var(--r-md)', fontSize:'var(--text-sm)', fontWeight:600, cursor:'pointer',
                      background:T.raised, border:`1px solid ${T.lineStrong}`, color:T.ink2, padding:'0 10px',
                      appearance:'auto', outline:'none' }}>
                    {[3,4,5,6,7,8,9,10].map(n => (
                      <option key={n} value={n}>{n} photos</option>
                    ))}
                  </select>
                </div>

                {/* Reference photo */}
                <div>
                  <label style={{ display:'block', fontSize:'var(--text-xs)', fontWeight:700, letterSpacing:'.07em', textTransform:'uppercase', color:T.ink2, marginBottom:4 }}>
                    Reference Photo <span style={{ fontWeight:400, textTransform:'none', letterSpacing:0, fontSize:'var(--text-xs)', color:T.ink3 }}>optional</span>
                  </label>
                  <p style={{ fontSize:'var(--text-xs)', color:T.ink3, lineHeight:1.4, marginBottom:10 }}>Sets the visual style anchor for the sequence.</p>
                  {creativeAnchor ? (
                    <div style={{ position:'relative', borderRadius:'var(--r-md)', overflow:'hidden', border:`2px solid ${T.mark}`, cursor:'pointer', boxShadow:`0 0 0 3px ${T.markDim}` }}
                      onClick={()=>setCreativeAnchor(null)} title="Click to remove">
                      <img src={thumbUrl(creativeAnchor)} alt="" style={{ width:'100%', aspectRatio:'3/2', objectFit:'cover', display:'block' }}/>
                      <div className="t-label absolute left-1 top-1 rounded-sm px-1" style={{ background:T.mark, color:T.well }}>ANCHOR</div>
                      <div style={{ position:'absolute', top:6, right:6, background:T.scrim, backdropFilter:'blur(4px)', borderRadius:'var(--r-sm)', padding:'3px 8px', fontSize:'var(--text-xs)', color:T.ink, fontWeight:600 }}>✕ remove</div>
                    </div>
                  ) : (
                    <div style={{ height:72, border:`2px dashed ${T.lineStrong}`, borderRadius:'var(--r-md)', display:'flex', alignItems:'center', justifyContent:'center', gap:7, color:T.ink3, fontSize:'var(--text-sm)' }}>
                      <Wand2 size={14} strokeWidth={1.5}/>
                      <span>Click a photo below to set anchor</span>
                    </div>
                  )}
                </div>

                {/* Photo picker grid */}
                {sortedPhotos.length > 0 && (
                  <div>
                    <p style={{ fontSize:'var(--text-xs)', color:T.ink3, marginBottom:8 }}>{sortedPhotos.length} photos · sorted by grade · click to anchor</p>
                    <div style={{ display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap:4 }}>
                      {sortedPhotos.map(p => {
                        const isAnchor = p.path===creativeAnchor;
                        const dc = gc(p.grade);
                        return (
                          <button key={p.id} onClick={()=>setCreativeAnchor(isAnchor?null:p.path)}
                            style={{ position:'relative', aspectRatio:'3/2', padding:0, border:'none', borderRadius:'var(--r-sm)', overflow:'hidden', cursor:'pointer',
                              outline: isAnchor?`2px solid ${T.mark}`:`1px solid ${T.line}`, outlineOffset:isAnchor?2:0,
                              transform:isAnchor?'scale(1.05)':'scale(1)', transition:'transform .12s, outline .12s' }}>
                            <img src={thumbUrl(p.path)} alt="" loading="eager" style={{ width:'100%', height:'100%', objectFit:'cover', display:'block' }}/>
                            <div style={{ position:'absolute', bottom:0, left:0, right:0, height:14, background:`linear-gradient(transparent, ${T.scrim})`, display:'flex', alignItems:'center', justifyContent:'flex-end', padding:'0 4px' }}>
                              <span style={{ fontSize:'var(--text-xs)', fontWeight:700, color:T.ink, fontVariantNumeric:'tabular-nums' }}>{Math.round(p.score*100)}</span>
                            </div>
                            {isAnchor && (
                              <div style={{ position:'absolute', inset:0, background:T.markDim, display:'flex', alignItems:'center', justifyContent:'center' }}>
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={T.ink} strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20,6 9,17 4,12"/></svg>
                              </div>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}

              </div>

              {/* Build controls — pinned to the bottom of the config column. */}
              <div className="shrink-0 border-t border-line px-4 py-3">
                {/* What each mode is FOR, in the user's terms. "Auto / Story /
                    Competition" alone made you pick before knowing the
                    difference; the line underneath says what you get. */}
                <div className="mb-2 flex flex-col gap-2">
                  <Segmented
                    className="w-full [&>button]:flex-1"
                    value={seqMode === 'director' ? 'story' : seqMode}
                    onChange={(m) => setSeqMode(m)}
                    options={[
                      { value: 'story',       label: 'Story' },
                      { value: 'competition', label: 'Contest' },
                      { value: 'auto',        label: 'Auto' },
                    ]}
                  />
                  <p className="text-xs text-ink-3">
                    {seqMode === 'competition'
                      ? 'Picks your strongest single frames — no repeats of the same look. For entries and portfolio reviews.'
                      : seqMode === 'auto'
                      ? 'Reads the set and chooses whichever approach fits it best.'
                      : 'Orders photos so they read like a story: an opening, a turn, a close.'}
                  </p>
                </div>

                {photos.length === 0 && (
                  <p className="mb-2 text-center text-xs text-ink-3">
                    Grade a folder first — there are no photos to work with yet.
                  </p>
                )}

                <Button
                  variant="solid"
                  disabled={!canGenerate}
                  onClick={handleRunCreativeDirection}
                  className="w-full"
                  icon={creativeLoading ? undefined : <Wand2 size={13}/>}
                >
                  {creativeLoading
                    ? 'Choosing photos…'
                    : hasResults ? 'Build it again'
                    : seqMode === 'competition' ? 'Pick my strongest'
                    : seqMode === 'auto' ? 'Build a set'
                    : 'Build the story'}
                </Button>

                {usedCount > 0 && (
                  <button onClick={handleClearUsed}
                    className="mt-2 w-full cursor-pointer border-0 bg-transparent py-1 text-center text-xs text-ink-3 transition-colors duration-fast ease hover:text-ink-2">
                    Put back <span className="t-num">{usedCount}</span> set-aside photos
                  </button>
                )}
              </div>
            </div>

            {/* ── Right results panel ──────────────────────────────── */}
            <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden' }}>

              {/* Progress bar (only while loading) */}
              {creativeLoading && (
                <div style={{ flexShrink:0, padding:'12px 20px', borderBottom:`1px solid ${T.line}`, background:T.surface }}>
                  <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:8 }}>
                    <div style={{width:11,height:11,border:`2px solid ${T.ink3}`,borderTopColor:'transparent',borderRadius:'var(--r-round)',animation:'spin .8s linear infinite',flexShrink:0}}/>
                    <span style={{fontSize:'var(--text-sm)',color:T.ink2,fontWeight:500}}>{creativeStage||'Building sequence…'}</span>
                    <span style={{marginLeft:'auto',fontSize:'var(--text-sm)',color:T.ink3,fontVariantNumeric:'tabular-nums'}}>{Math.round(creativeProgress*100)}%</span>
                  </div>
                  <div style={{height:3,background:T.lineStrong,borderRadius:'var(--r-sm)',overflow:'hidden'}}>
                    <div style={{height:'100%',width:`${Math.round(creativeProgress*100)}%`,background:T.ink3,borderRadius:'var(--r-sm)',transition:'width .4s cubic-bezier(.2,0,0,1)'}}/>
                  </div>
                </div>
              )}

              {hasResults ? (
                <>
                  {/* Results toolbar */}
                  <div style={{flexShrink:0, display:'flex', alignItems:'center', justifyContent:'space-between', padding:'10px 20px', borderBottom:`1px solid ${T.line}`, background:T.surface}}>
                    <div style={{display:'flex', alignItems:'center', gap:10}}>
                      <span style={{fontSize:'var(--text-sm)', fontWeight:700}}>Story sequence</span>
                      <span style={{fontSize:'var(--text-xs)', color:T.ink3, background:T.raised, borderRadius:'var(--r-sm)', padding:'2px 8px'}}>{successResults.length} images</span>
                      {creativeResults.some((r:any)=>!r.success) && (
                        <span style={{fontSize:'var(--text-xs)', color:T.gradeWeak, cursor:'default'}}
                          title={creativeResults.filter((r:any)=>!r.success).map((r:any)=>`${(r.source_path??'').split(/[\\/]/).pop()}: ${r.error??'failed'}`).join('\n')}>
                          {creativeResults.filter((r:any)=>!r.success).length} failed ⓘ
                        </span>
                      )}
                      {creativeFallback && (
                        <span style={{fontSize:'var(--text-xs)', color:T.gradeWeak, cursor:'default'}}
                          title={`No art direction ran — ${creativeFallback}. These are the highest-scoring frames in score order, not a curated sequence.`}>
                          sorted by score ⓘ
                        </span>
                      )}
                    </div>
                    <div style={{display:'flex', alignItems:'center', gap:8}}>
                      {!creativeLoading && (
                        <button disabled={sequenceSaving} onClick={handleSaveSequence}
                          style={{display:'flex', alignItems:'center', gap:5, fontSize:'var(--text-sm)', fontWeight:600, padding:'4px 12px', borderRadius:'var(--r-md)',
                            cursor:sequenceSaving?'wait':'pointer', background:'transparent', border:`1px solid ${T.lineStrong}`, color:T.ink2, transition:'all .15s'}}>
                          <Download size={11}/>{sequenceSaving?'Saving…':'Save Sequence'}
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Sequence grid — landscape cards, 2–3 per row */}
                  <div style={{flex:1, overflowY:'auto', padding:'18px 20px'}}>
                    <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(240px, 1fr))', gap:14}}>
                      {successResults.map((r:any, i:number) => {
                        const slot  = r.slot ?? r.params?.role ?? `Frame ${i+1}`;
                        const sc    = slotColor(slot);
                        const fname = (r.source_path??'').split(/[\\/]/).pop()??'';
                        const photoScore = photos.find((p:any)=>p.path===r.source_path)?.score;
                        return (
                          <div key={i} style={{borderRadius:'var(--r-md)', overflow:'hidden', border:`1px solid ${T.line}`, background:T.surface, display:'flex', flexDirection:'column', boxShadow:`0 2px 12px ${T.well}`}}>
                            {/* Slot header */}
                            <div style={{padding:'8px 12px', background:T.raised, borderBottom:`2px solid ${sc}`, display:'flex', alignItems:'center', gap:8}}>
                              <span style={{fontSize:'var(--text-xs)', fontWeight:800, letterSpacing:'.12em', color:sc, textTransform:'uppercase', flex:1}}>{slot}</span>
                              <span style={{fontSize:'var(--text-xs)', color:T.ink3, fontWeight:600, background:T.raisedHover, borderRadius:'var(--r-sm)', padding:'1px 6px'}}>
                                {i+1}/{successResults.length}
                              </span>
                            </div>
                            {/* Photo — landscape 4:3 */}
                            <div style={{position:'relative', aspectRatio:'4/3', overflow:'hidden', background:T.ground}}>
                              <img src={photoUrl(r.source_path ?? r.output_path)} alt="" loading="eager" decoding="async"
                                style={{width:'100%', height:'100%', objectFit:'cover', display:'block'}}/>
                              <div style={{position:'absolute', inset:0, pointerEvents:'none',
                                background:`linear-gradient(to bottom, transparent 55%, ${T.scrim} 100%)`}}/>
                              <a href={photoUrl(r.output_path ?? r.source_path)} download={fname} onClick={e=>e.stopPropagation()}
                                style={{position:'absolute', top:8, right:8, background:T.scrim, backdropFilter:'blur(4px)', borderRadius:'var(--r-sm)', padding:'5px 8px', fontSize:'var(--text-xs)', color:T.ink, textDecoration:'none', display:'flex', alignItems:'center', gap:3, fontWeight:600, opacity:.85}}>
                                <Download size={9}/>
                              </a>
                              {photoScore!=null && (
                                <div style={{position:'absolute', bottom:8, right:10, display:'flex', alignItems:'center', gap:3,
                                  background:T.scrim, backdropFilter:'blur(6px)', borderRadius:'var(--r-sm)', padding:'2px 8px'}}>
                                  <div style={{width:5, height:5, borderRadius:'var(--r-round)', background:sc}}/>
                                  <span style={{fontSize:'var(--text-sm)', fontWeight:800, color:T.ink, fontVariantNumeric:'tabular-nums'}}>{Math.round(photoScore*100)}</span>
                                </div>
                              )}
                            </div>
                            {/* Filename */}
                            <div style={{padding:'8px 12px'}}>
                              <span style={{fontSize:'var(--text-xs)', color:T.ink3, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', display:'block'}} title={fname}>{fname}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </>
              ) : (
                /* Empty state */
                <div style={{flex:1, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:18, color:T.ink3, padding:40}}>
                  <Wand2 size={44} strokeWidth={1} style={{opacity:.3}}/>
                  <div style={{textAlign:'center', maxWidth:360}}>
                    <p style={{fontSize:'var(--text-md)', fontWeight:700, color:T.ink2, marginBottom:10}}>No sequence yet</p>
                    <p style={{fontSize:'var(--text-sm)', lineHeight:1.75, margin:0, color:T.ink3}}>
                      Write a mood brief on the left,<br/>
                      optionally pick a reference photo,<br/>
                      then press <strong className="text-ink">Build story sequence</strong>.
                    </p>
                    {photos.length===0 && (
                      <p style={{fontSize:'var(--text-sm)', color:T.gradeWeak, marginTop:14}}>Grade a folder first to load photos.</p>
                    )}
                  </div>
                </div>
              )}

            </div>
          </div>
          );
        })()
      ) : null}

      {/* ── Status bar ─────────────────────────────────────────── */}
      <div className="flex h-6 shrink-0 items-center gap-4 border-t border-line bg-surface px-3">
        <span className="t-num flex-1 truncate text-xs text-ink-2">
          {sel ? sel.path.split(/[\\/]/).pop() : 'Open a folder to begin'}
        </span>
        <div className="flex shrink-0 gap-3">
          {[['← →','Navigate'],['1–5','Rate'],['G','Grid'],['E','Loupe']].map(([k, a]) => (
            <span key={k} className="flex items-center gap-1 text-xs text-ink-3">
              <kbd className="t-num rounded-sm border border-line-strong bg-raised px-1 text-xs text-ink-2">{k}</kbd>
              {a}
            </span>
          ))}
        </div>
      </div>

      {/* ── Folder browser modal ────────────────────────────────── */}
      {showBrowser && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-scrim p-4">
          <div className="flex h-[82vh] w-full max-w-[640px] flex-col overflow-hidden rounded-md border border-line-strong bg-surface shadow-lg">
            <div className="flex shrink-0 items-center justify-between border-b border-line px-4 py-3">
              <span className="text-md font-semibold text-ink">Choose a photo folder</span>
              <button onClick={() => setShowBrowser(false)} aria-label="Close"
                className="cursor-pointer rounded-sm border-0 bg-transparent p-1 text-ink-3 transition-colors duration-fast ease hover:bg-raised hover:text-ink">
                <X size={16}/>
              </button>
            </div>
            <div className="flex shrink-0 items-center gap-2 border-b border-line bg-well px-3 py-2">
              <Button size="sm" onClick={goUp} icon={<ArrowUp size={11}/>}>Up</Button>
              <span className="t-num flex-1 truncate rounded-sm border border-line-strong bg-raised px-2 py-1 text-xs text-ink-2">
                {bPath}
              </span>
              <Button
                size="sm"
                variant="solid"
                onClick={async () => {
                  try {
                    if (browserMode === 'add') {
                      const toAdd = bSelFolders.size ? Array.from(bSelFolders) : [bPath];
                      for (const nf of toAdd) await handleAddFolder(nf);
                    } else {
                      setFolder(bPath); setPhotos([]); setSelId(null); setFolders([]);
                    }
                  } catch (err) { /* non-blocking */ }
                  setShowBrowser(false);
                  setBSelFolders(new Set());
                }}
                disabled={bImages.length===0}
                title={bImages.length === 0 ? 'This folder holds no images' : undefined}>
                {browserMode === 'add' ? 'Add' : 'Use folder'}
                {bImages.length > 0 && <span className="t-num ml-1 opacity-70">{bImages.length}</span>}
              </Button>
            </div>
            <div className="flex flex-1 overflow-hidden">
              <div className="flex w-sidebar shrink-0 flex-col gap-px overflow-y-auto border-r border-line bg-well p-2">
                <p className="t-label mb-1 px-2">Quick access</p>
                {([
                  { label:'Desktop',   path:'C:\\Users\\Nicky Tuason\\Desktop' },
                  { label:'Pictures',  path:'C:\\Users\\Nicky Tuason\\Pictures' },
                  { label:'Downloads', path:'C:\\Users\\Nicky Tuason\\Downloads' },
                  { label:'Documents', path:'C:\\Users\\Nicky Tuason\\Documents' },
                  { label:'C:\\',      path:'C:\\' },
                ]).map(loc => (
                  <button key={loc.path} onClick={() => { setBPath(loc.path); loadBrowser(loc.path); }}
                    className={cn(
                      'truncate rounded-sm border-0 px-2 py-1 text-left text-sm',
                      'cursor-pointer transition-colors duration-fast ease',
                      bPath === loc.path
                        ? 'bg-raised-hover text-ink'
                        : 'bg-transparent text-ink-3 hover:bg-raised hover:text-ink-2',
                    )}>
                    {loc.label}
                  </button>
                ))}
              </div>
              <div className="flex-1 overflow-y-auto p-4">
                {bLoading ? (
                  <div className="flex h-full items-center justify-center text-ink-3">
                    <span className="text-sm">Reading folder…</span>
                  </div>
                ) : bFolders.length===0 && bImages.length===0 ? (
                  <div className="flex h-full flex-col items-center justify-center gap-2 text-ink-3">
                    <FolderOpen size={28} strokeWidth={1.5}/>
                    <p className="text-sm">Nothing here</p>
                    <p className="text-xs text-ink-4">No folders or images in this location.</p>
                  </div>
                ) : (
                  <>
                    {bFolders.length > 0 && (
                      <div className="mb-6">
                        <p className="t-label mb-2">Folders <span className="t-num">{bFolders.length}</span></p>
                        <div className="grid gap-1" style={{ gridTemplateColumns:'repeat(auto-fill, minmax(150px,1fr))' }}>
                          {bFolders.map((f, idx) => (
                            <button key={f} onClick={(e) => handleBrowserFolderClick(e as any, f, idx)}
                              className={cn(
                                'flex cursor-pointer items-center gap-2 rounded-sm border px-3 py-2 text-left',
                                'transition-colors duration-fast ease',
                                bSelFolders.has(f)
                                  ? 'border-mark bg-raised text-ink'
                                  : 'border-line-strong bg-raised text-ink-2 hover:bg-raised-hover hover:text-ink',
                              )}>
                              <FolderOpen size={13} className="shrink-0 text-ink-3"/>
                              <span className="truncate text-sm">{f.split(/[\\/]/).pop()}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {bImages.length > 0 && (
                      <div>
                        <p className="t-label mb-2">Images <span className="t-num">{bImages.length}</span></p>
                        <div className="flex flex-wrap gap-1">
                          {bImages.slice(0,30).map(img => (
                            <div key={img} className="overflow-hidden rounded-sm border border-line bg-well">
                              <img src={thumbUrl(img)} className="block h-thumb w-auto max-w-none" loading="lazy" alt=""/>
                            </div>
                          ))}
                          {bImages.length > 30 && (
                            <div className="flex h-thumb items-center justify-center rounded-sm border border-line bg-raised px-3">
                              <span className="t-num text-xs text-ink-3">+{bImages.length-30} more</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
