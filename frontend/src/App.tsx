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

/* ── Design tokens ─────────────────────────────────────────────── */
const C = {
  bg:     '#0a0a0d',
  surf:   '#111114',
  surf2:  '#18181e',
  surf3:  '#1e1e27',
  border: '#1c1c24',
  bdr2:   '#252535',
  text:   '#e8e8ed',
  text2:  '#8a8a9a',
  text3:  '#44445a',
  accent: 'oklch(64% .19 248)',
  aLow:   'oklch(64% .19 248 / .12)',
  aBdr:   'oklch(64% .19 248 / .3)',
  strong: 'oklch(65% .17 148)',
  sLow:   'oklch(65% .17 148 / .14)',
  mid:    'oklch(70% .17 72)',
  mLow:   'oklch(70% .17 72 / .14)',
  weak:   'oklch(58% .18 18)',
  wLow:   'oklch(58% .18 18 / .14)',
};

function gc(g: string) {
  if (g?.includes('Strong')) return C.strong;
  if (g?.includes('Mid'))    return C.mid;
  if (g?.includes('Weak'))   return C.weak;
  return C.text3;
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
const tierColor = (t: RegionTier) => t === 'strong' ? C.strong : t === 'fix' ? C.weak : C.mid;
const tierIcon  = (t: RegionTier) => t === 'strong' ? '✓' : t === 'fix' ? '!' : '◐';
// Vivid thermal palette for the soft-glow heatmap (kept distinct from theme tokens).
const tierHeat  = (t: RegionTier) => t === 'strong' ? '#3fb950' : t === 'fix' ? '#f85149' : '#d8a657';
function gLow(g: string) {
  if (g?.includes('Strong')) return C.sLow;
  if (g?.includes('Mid'))    return C.mLow;
  if (g?.includes('Weak'))   return C.wLow;
  return 'transparent';
}
function gl(g: string) {
  if (g?.includes('Strong')) return 'Strong';
  if (g?.includes('Mid'))    return 'Mid';
  if (g?.includes('Weak'))   return 'Weak';
  return 'Pending';
}
function gIcon(g: string) {
  if (g?.includes('Strong')) return '✅';
  if (g?.includes('Mid'))    return '⚠️';
  if (g?.includes('Weak'))   return '❌';
  return '';
}

const _SLOGANS: Array<[RegExp, string]> = [
  [/scan|found.*image|new image/i,        "Pulling the contact sheet…"],
  [/blur|early.exit|laplacian/i,          "Culling the camera-shake casualties…"],
  [/siglip|encod/i,                       "Reading the light in every frame…"],
  [/duplicat/i,                           "Picking the best frame from each burst…"],
  [/loading qwen|qwen.*load/i,            "The photo editor is pulling up a chair…"],
  [/qwen|vlm grad|vision grad/i,          "Studying composition, moment, and story…"],
  [/iqa|topiq|maniqa|technical.*scor/i,   "Running the darkroom technical check…"],
  [/luminance|light.*stat/i,              "Measuring the exposure…"],
  [/specvlm|clip.*sim/i,                  "Comparing against the reference portfolio…"],
  [/personal|head.*scor/i,                "Recalling your editorial eye…"],
  [/gemma|spatial|second/i,               "Second shooter weighing in…"],
  [/sequenc|sort|bucket|calibrat|thresh/i,"Building the selects…"],
  [/archetype|fusion|fus/i,               "Matching each frame to its genre…"],
  [/deduplic|similar/i,                   "One frame per moment — killing your darlings…"],
  [/persona|preference/i,                 "Tuning to your shooting style…"],
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
  const w = Math.round(h * 1.5);
  return (
    <button
      data-sel={isSel ? '1' : '0'}
      onClick={() => onSelect(p.id)}
      style={{
        flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 2,
        width: w, padding: 2, borderRadius: 3, cursor: 'pointer',
        background: isSel ? C.surf3 : 'transparent',
        outline: isSelected ? `2px solid ${C.accent}` : isSel ? '2px solid rgba(255,255,255,.5)' : '2px solid transparent',
        outlineOffset: 0, border: 'none',
      }}
    >
      <div style={{ position: 'relative', width: w - 4, height: h - 4, overflow: 'hidden', borderRadius: 2, background: C.bg, flexShrink: 0 }}>
        <img src={thumbUrl(p.path)} alt="" decoding="async" loading="lazy"
          style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}/>
        {isUsed && (
          <div style={{ position: 'absolute', top: 3, left: 3, background: 'rgba(0,0,0,.75)', backdropFilter: 'blur(4px)', borderRadius: 3, padding: '1px 4px', display: 'flex', alignItems: 'center', gap: 2 }}>
            <Flag size={7} style={{ color: C.accent, flexShrink: 0 }}/>
          </div>
        )}
        {isSelected && (
          <div style={{ position: 'absolute', top: 3, right: 3, width: 12, height: 12, borderRadius: 3, background: C.accent, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="7" height="7" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20,6 9,17 4,12"/></svg>
          </div>
        )}
        {p.grade !== 'Pending' && gc(p.grade) !== C.text3 && (
          <div style={{ position:'absolute', bottom:3, left:3, display:'flex', alignItems:'center', gap:2 }}>
            <div style={{ width:6, height:6, borderRadius:'50%', background:gc(p.grade), boxShadow:`0 0 5px ${gc(p.grade)}99` }}/>
            {p.has_annotations && (
              <svg width="7" height="7" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
              </svg>
            )}
          </div>
        )}
      </div>
      {showFn && (
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', width: w - 4, gap: 2 }}>
          <span style={{ fontSize: 8.5, color: isSel ? C.text2 : C.text3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontFamily: "'SF Mono',monospace", flex:1 }}>
            {(p.path.split(/[\\/]/).pop() ?? '').replace(/\.[^.]+$/, '')}
          </span>
          {p.stars > 0 && (
            <div style={{ display:'flex', gap:0.5, flexShrink:0 }}>
              {[1,2,3,4,5].map(n => (
                <svg key={n} width="5" height="5" viewBox="0 0 24 24"
                  fill={n <= p.stars ? 'oklch(70% .18 72)' : 'none'}
                  stroke={n <= p.stars ? 'oklch(70% .18 72)' : C.text3} strokeWidth="2">
                  <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
                </svg>
              ))}
            </div>
          )}
        </div>
      )}
    </button>
  );
});

/* ── Star Rating ────────────────────────────────────────────────── */
function StarRating({ stars, onSet, size = 22, gap = 4 }: { stars: number; onSet: (n: number) => void; size?: number; gap?: number }) {
  const [hover, setHover] = useState(0);
  const display = hover || stars;
  return (
    <div style={{ display:'flex', alignItems:'center', gap }} onMouseLeave={() => setHover(0)}>
      {[1,2,3,4,5].map(n => (
        <button key={n} onMouseEnter={() => setHover(n)} onClick={() => onSet(stars === n ? 0 : n)}
          style={{ padding:4, cursor:'pointer', display:'flex', lineHeight:1, background:'none', border:'none', flexShrink:0 }}>
          <svg width={size} height={size} viewBox="0 0 24 24"
            fill={n <= display ? 'oklch(70% .18 72)' : 'oklch(30% .04 72)'}
            stroke="none"
            style={{ transition:'fill .2s ease' }}>
            <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
          </svg>
        </button>
      ))}
    </div>
  );
}

/* ── EXIF Block ──────────────────────────────────────────────────── */
function ExifBlock({ exif }: { exif: any }) {
  if (!exif || !Object.keys(exif).length) return (
    <p style={{ fontSize:12, color:C.text3, lineHeight:1.7 }}>No EXIF data available for this photo.</p>
  );
  const ORDER: [string, string][] = [
    ['camera',        'Camera'],
    ['lens',          'Lens'],
    ['focal',         'Focal Length'],
    ['focal_35mm',    '35mm Equiv.'],
    ['aperture',      'Aperture'],
    ['shutter',       'Shutter'],
    ['iso',           'ISO'],
    ['ev',            'Exp. Bias'],
    ['program',       'Mode'],
    ['metering',      'Metering'],
    ['white_balance', 'White Balance'],
    ['flash',         'Flash'],
    ['date',          'Date'],
    ['time',          'Time'],
    ['gps',           'GPS'],
  ];
  const rows = ORDER.filter(([k]) => exif[k] != null).map(([k, label]) => [label, String(exif[k])] as [string,string]);
  return (
    <div style={{ display:'flex', flexDirection:'column', gap:0 }}>
      <p style={{ fontSize:11, fontWeight:700, letterSpacing:'.08em', textTransform:'uppercase', color:C.text3, marginBottom:8 }}>EXIF Data</p>
      {rows.map(([k, v]) => (
        <div key={k} style={{ display:'flex', justifyContent:'space-between', alignItems:'center', padding:'6px 0', borderBottom:`1px solid ${C.border}` }}>
          <span style={{ fontSize:12, color:C.text3, fontWeight:500 }}>{k}</span>
          <span style={{ fontSize:12, color:C.text, fontWeight:600, fontVariantNumeric:'tabular-nums', fontFamily:"'SF Mono',monospace", textAlign:'right', maxWidth:'60%', wordBreak:'break-word' }}>{v}</span>
        </div>
      ))}
    </div>
  );
}

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
    <div style={{ position:'fixed', inset:0, zIndex:500, background:'rgba(0,0,0,.75)', backdropFilter:'blur(8px)', display:'flex', alignItems:'center', justifyContent:'center' }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div style={{ background:C.surf, border:`1px solid ${C.bdr2}`, borderRadius:12, width:560, maxHeight:'80vh', display:'flex', flexDirection:'column', boxShadow:'0 24px 80px rgba(0,0,0,.8)', overflow:'hidden', animation:'slideUp .3s cubic-bezier(.2,0,0,1)' }}>
        <div style={{ display:'flex', alignItems:'center', padding:'14px 18px', borderBottom:`1px solid ${C.border}`, flexShrink:0 }}>
          <div style={{ flex:1 }}>
            <p style={{ fontSize:15, fontWeight:700, color:C.text }}>Export Photos</p>
            <p style={{ fontSize:12, color:C.text3, marginTop:2 }}>{photos.length} photo{photos.length !== 1 ? 's' : ''}{filterGrade ? ` · ${filterGrade} only` : ''}</p>
          </div>
          <button onClick={onClose} style={{ color:C.text3, display:'flex', padding:6, borderRadius:6, cursor:'pointer' }}>
            <X size={13}/>
          </button>
        </div>
        <div style={{ flex:1, overflow:'auto', padding:'10px 18px' }}>
          {photos.map(p => (
            <div key={p.id} style={{ display:'flex', alignItems:'center', gap:10, padding:'7px 0', borderBottom:`1px solid ${C.border}` }}>
              <img src={thumbUrl(p.path)} alt="" style={{ width:48, height:32, objectFit:'cover', borderRadius:3, flexShrink:0, display:'block' }}/>
              <div style={{ flex:1, minWidth:0 }}>
                <p style={{ fontSize:13, fontWeight:600, color:C.text, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{p.path.split(/[\\/]/).pop()}</p>
                <p style={{ fontSize:11, color:C.text3, marginTop:1, fontFamily:"'SF Mono',monospace" }}>
                  {[p.exif?.camera, p.exif?.aperture, p.exif?.shutter, p.exif?.iso ? `ISO ${p.exif.iso}` : null].filter(Boolean).join(' · ')}
                </p>
              </div>
              <button onClick={() => handleDownload(p)}
                style={{ display:'flex', alignItems:'center', gap:4, padding:'4px 9px', borderRadius:6, background:C.surf2, border:`1px solid ${C.bdr2}`, color:C.text2, fontSize:12, fontWeight:600, cursor:'pointer', flexShrink:0 }}>
                <Download size={10}/>
              </button>
            </div>
          ))}
        </div>

        {/* XMP sidecar section */}
        <div style={{ padding:'10px 18px', borderTop:`1px solid ${C.border}`, background:C.surf2, flexShrink:0 }}>
          <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', gap:10 }}>
            <div style={{ minWidth:0 }}>
              <p style={{ fontSize:12, fontWeight:600, color:C.text2 }}>XMP Sidecars</p>
              <p style={{ fontSize:11, color:C.text3, marginTop:1 }}>
                {xmpState === 'idle' && 'Write .xmp files next to each photo — readable by Lightroom & Capture One'}
                {xmpState === 'busy' && 'Writing sidecars…'}
                {xmpState === 'done' && `✓ ${xmpCount} sidecar${xmpCount !== 1 ? 's' : ''} written next to your photos`}
                {xmpState === 'error' && '✕ Export failed — check the server log'}
              </p>
            </div>
            <button onClick={handleExportXmp} disabled={xmpState === 'busy'}
              style={{ display:'flex', alignItems:'center', gap:6, padding:'6px 14px', borderRadius:7, flexShrink:0,
                background: xmpState === 'done' ? C.sLow : C.surf3,
                border:`1px solid ${xmpState === 'done' ? 'oklch(65% .17 148 / .35)' : C.bdr2}`,
                color: xmpState === 'done' ? C.strong : C.text2,
                fontSize:12, fontWeight:700, cursor: xmpState === 'busy' ? 'wait' : 'pointer', transition:'all .25s cubic-bezier(.2,0,0,1)' }}>
              {xmpState === 'busy'
                ? <><span style={{ width:10, height:10, borderRadius:'50%', border:`1.5px solid ${C.accent}`, borderTopColor:'transparent', animation:'spin .8s linear infinite', display:'inline-block' }}/> Writing…</>
                : xmpState === 'done' ? 'Done' : 'Export XMP'}
            </button>
          </div>
        </div>

        <div style={{ padding:'12px 18px', borderTop:`1px solid ${C.border}`, display:'flex', justifyContent:'flex-end', gap:8, flexShrink:0 }}>
          <button onClick={onClose} style={{ padding:'7px 16px', borderRadius:7, background:C.surf2, border:`1px solid ${C.bdr2}`, color:C.text2, fontSize:13, fontWeight:600, cursor:'pointer' }}>Cancel</button>
          <button onClick={handleDownloadAll}
            style={{ display:'flex', alignItems:'center', gap:6, padding:'7px 18px', borderRadius:7, background:C.accent, border:'none', color:'#fff', fontSize:13, fontWeight:700, cursor:'pointer' }}>
            <Download size={11}/> Download All ({photos.length})
          </button>
        </div>
      </div>
    </div>
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
      {/* Toolbar */}
      <div style={{ flexShrink:0, height:36, display:'flex', alignItems:'center', justifyContent:'space-between', padding:'0 14px', background:C.surf, borderBottom:`1px solid ${C.border}` }}>
        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          <button onClick={() => { setSelectMode(!selectMode); setSelectedIds(new Set()); }}
            style={{ display:'flex', alignItems:'center', gap:5, padding:'4px 10px', borderRadius:6, fontSize:12, fontWeight:700, cursor:'pointer', background:selectMode ? C.aLow : 'transparent', border:`1px solid ${selectMode ? C.aBdr : C.bdr2}`, color:selectMode ? C.accent : C.text3, transition:'all .25s cubic-bezier(.2,0,0,1)' }}>
            <CheckSquare size={11}/>{selectMode ? `Select (${selectedIds.size})` : 'Select'}
          </button>
          {selectMode && selectedIds.size > 0 && (
            <button onClick={() => setSelectedIds(new Set())}
              style={{ fontSize:11, color:C.text3, padding:'3px 7px', borderRadius:5, border:`1px solid ${C.bdr2}`, background:C.surf2, cursor:'pointer' }}>
              ✕ Clear
            </button>
          )}
        </div>
        <span style={{ fontSize:11, color:C.text3 }}>{photos.length} photos</span>
      </div>

      {/* Grid */}
      <div style={{ flex:1, overflow:'auto', background:C.bg, padding:10 }}>
        <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(180px, 1fr))', gap:6 }}>
          {photos.map(p => {
            const isChecked = selectedIds.has(p.id);
            const isUsed    = usedPaths.has(p.path);
            const isCurrent = p.id === selId && !selectMode;
            return (
              <button key={p.id} onClick={() => selectMode ? toggleSelect(p.id) : onSelect(p.id)}
                style={{
                  position:'relative', display:'flex', flexDirection:'column',
                  background:'transparent', borderRadius:4, overflow:'hidden', cursor:'pointer',
                  outline: isChecked ? `2px solid ${C.accent}` : isCurrent ? `2px solid rgba(255,255,255,.5)` : `2px solid transparent`,
                  outlineOffset:1, padding:0, border:'none', transition:'outline .2s ease',
                  contentVisibility:'auto', containIntrinsicSize:'180px 120px',
                }}>
                <div style={{ position:'relative', width:'100%', aspectRatio:'3/2', background:C.surf2, overflow:'hidden' }}>
                  <img src={thumbUrl(p.path)} alt="" decoding="async" loading="lazy"
                    style={{ width:'100%', height:'100%', objectFit:'cover', display:'block', opacity: selectMode && !isChecked ? 0.55 : 1, transition:'opacity .15s' }}/>
                  {selectMode && (
                    <div style={{ position:'absolute', top:6, left:6, width:16, height:16, borderRadius:4, background:isChecked ? C.accent : 'rgba(0,0,0,.6)', border:`1.5px solid ${isChecked ? C.accent : 'rgba(255,255,255,.4)'}`, display:'flex', alignItems:'center', justifyContent:'center', transition:'all .25s cubic-bezier(.2,0,0,1)' }}>
                      {isChecked && <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20,6 9,17 4,12"/></svg>}
                    </div>
                  )}
                  {isUsed && (
                    <div style={{ position:'absolute', top:5, right:5, background:'rgba(0,0,0,.75)', backdropFilter:'blur(4px)', borderRadius:3, padding:'1px 5px', display:'flex', alignItems:'center', gap:2 }}>
                      <Flag size={7} style={{ color:C.accent, flexShrink:0 }}/>
                      <span style={{ fontSize:9, fontWeight:700, color:C.accent }}>USED</span>
                    </div>
                  )}
                  {p.grade !== 'Pending' && p.score > 0 && (
                    <div style={{ position:'absolute', bottom:5, left:5,
                      background:'rgba(0,0,0,.68)', backdropFilter:'blur(8px)',
                      borderRadius:5, padding:'3px 7px', display:'flex', alignItems:'center', gap:4,
                      border:`1px solid ${gc(p.grade)}44`, pointerEvents:'none' }}>
                      <div style={{ width:6, height:6, borderRadius:'50%', background:gc(p.grade), flexShrink:0 }}/>
                      {p.has_annotations && (
                        <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink:0 }}>
                          <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
                        </svg>
                      )}
                    </div>
                  )}
                </div>
                <div style={{ padding:'4px 6px', background:isChecked ? `oklch(64% .19 248 / .1)` : isCurrent ? C.surf3 : C.surf, display:'flex', alignItems:'center', gap:4 }}>
                  <span style={{ fontSize:10.5, color:isChecked ? C.accent : C.text2, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', fontFamily:"'SF Mono',monospace", flex:1 }}>
                    {(p.path.split(/[\\/]/).pop() ?? '').replace(/\.[^.]+$/, '')}
                  </span>
                  {p.stars > 0 && (
                    <div style={{ display:'flex', gap:1, flexShrink:0, marginLeft:4 }}>
                      {[1,2,3,4,5].map(n => (
                        <svg key={n} width="7" height="7" viewBox="0 0 24 24"
                          fill={n <= p.stars ? 'oklch(70% .18 72)' : 'none'}
                          stroke={n <= p.stars ? 'oklch(70% .18 72)' : C.text3} strokeWidth="2">
                          <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
                        </svg>
                      ))}
                    </div>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Selection action bar */}
      {selectMode && selectedIds.size > 0 && (
        <div style={{ position:'absolute', bottom:16, left:'50%', transform:'translateX(-50%)', display:'flex', alignItems:'center', gap:10, background:C.surf, border:`1px solid ${C.bdr2}`, borderRadius:12, padding:'10px 18px', boxShadow:'0 8px 40px rgba(0,0,0,.7)', backdropFilter:'blur(12px)', zIndex:50, whiteSpace:'nowrap', animation:'slideUp .3s cubic-bezier(.2,0,0,1)' }}>
          <span style={{ fontSize:14, fontWeight:700, color:C.text }}>{selectedIds.size} selected</span>
          <div style={{ width:1, height:16, background:C.bdr2 }}/>
          <button onClick={onCreateSequence}
            style={{ display:'flex', alignItems:'center', gap:6, padding:'6px 14px', borderRadius:8, background:C.accent, border:'none', color:'#fff', fontSize:13, fontWeight:700, cursor:'pointer' }}>
            <Layers size={11}/> Start Sequence
          </button>
          <button onClick={onAutoSequence}
            style={{ display:'flex', alignItems:'center', gap:6, padding:'6px 14px', borderRadius:8, background:C.surf2, border:`1px solid ${C.bdr2}`, color:C.text2, fontSize:13, fontWeight:600, cursor:'pointer' }}>
            <RefreshCw size={11}/> Auto
          </button>
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
    blur:    '#60a5fa',
    heatmap: '#fb923c',
    grid:    '#a78bfa',
  };
  while ((m = RE.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const typ   = m[1];
    const label = m[2];
    const col   = COLORS[typ] ?? '#60a5fa';
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
const FACTOR_COLORS: Record<string, string> = {
  blur:    '#60a5fa',
  heatmap: '#fb923c',
  grid:    '#a78bfa',
};

const _INK  = 'rgba(255,255,255,0.92)';
const _SH   = '0 0 8px rgba(0,0,0,1), 0 1px 3px rgba(0,0,0,.95)';
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
          <feDropShadow dx="0" dy="0" stdDeviation="2.5" floodColor="#000" floodOpacity="1"/>
        </filter>
      </defs>
      {factors.map((f: any, i: number) => {
        const [bx, by, bw, bh] = REGION_BOX[f.region] ?? REGION_BOX['full'];
        const isStrength = (f.impact ?? 0) > 0;
        const isWeakness = (f.impact ?? 0) < 0;
        const color = isStrength ? '#f5c842' : isWeakness ? '#ef4444' : (FACTOR_COLORS[f.type] ?? '#94a3b8');

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
                fill="none" stroke="#f5c842" strokeWidth="2.2"
                strokeOpacity="0.85" strokeDasharray="7 4"/>
            )}

            {/* Weakness: bold X-strike */}
            {isWeakness && (<>
              <line x1={`${rx + rw*0.06}%`} y1={`${ry + rh*0.06}%`}
                    x2={`${rx + rw*0.94}%`} y2={`${ry + rh*0.94}%`}
                stroke="#ef4444" strokeWidth="2.2" strokeOpacity="0.8"/>
              <line x1={`${rx + rw*0.94}%`} y1={`${ry + rh*0.06}%`}
                    x2={`${rx + rw*0.06}%`} y2={`${ry + rh*0.94}%`}
                stroke="#ef4444" strokeWidth="2.2" strokeOpacity="0.8"/>
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
        <span style={{ fontFamily:_MONO, fontSize:32, fontWeight:700, lineHeight:1,
          color:gradeColor, textShadow:_SH, letterSpacing:'-.01em' }}>{pct}</span>
        <span style={{ fontFamily:_MONO, fontSize:11, color:_INK, textShadow:_SH, opacity:.45 }}>/100</span>
        <span style={{ fontFamily:_MONO, fontSize:11, fontWeight:700, letterSpacing:'.1em',
          color:gradeColor, textShadow:_SH, marginLeft:4 }}>{gl(grade).toUpperCase()}</span>
      </div>
      {/* Aspect scores as pen-ruled tick lines */}
      <div style={{ display:'flex', flexDirection:'column', gap:5 }}>
        {aspects.map(([k, v]) => {
          const vpct = Math.round((v as number) * 100);
          const isTop = v === maxV;
          const isBot = v === minV && v !== maxV;
          const col = isTop ? C.strong : isBot ? C.weak : _INK;
          const filled = vpct * 0.76;
          return (
            <div key={k} style={{ display:'flex', alignItems:'center', gap:8 }}>
              <span style={{ fontFamily:_MONO, fontSize:9, color:col, textShadow:_SH,
                width:82, textAlign:'right', flexShrink:0, opacity:.85, letterSpacing:'.02em' }}>
                {k}
              </span>
              <svg width="80" height="9" style={{ flexShrink:0, overflow:'visible' }}>
                <line x1="0" y1="4.5" x2="76" y2="4.5" stroke={`${col}`} strokeWidth="0.75" opacity="0.25"/>
                <line x1="0" y1="4.5" x2={filled} y2="4.5" stroke={col} strokeWidth="1.5" strokeLinecap="round"/>
                <line x1={filled} y1="1" x2={filled} y2="8" stroke={col} strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              <span style={{ fontFamily:_MONO, fontSize:10, fontWeight:700, color:col,
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
          modelError = e?.message ?? 'Network error — is Ollama running?';
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

  /* lazy EXIF fetch — load when a photo is selected and has no EXIF yet */
  useEffect(() => {
    if (!sel || Object.keys(sel.exif || {}).length > 0) return;
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
      if (filterGrade) return gl(p.grade) === filterGrade && starsOk;
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
        if (!rawPhotos.length) notify("No images found in selected folder", "info");
        const ps = rawPhotos.map((p, i) => ({ id:`p-${i}`, path:p.path, grade:'Pending', score:0, breakdown:{}, critique:'', reasoning_log:'', is_verified:false, stars:0, exif:p.exif||{} }));
        setPhotos(ps);
        setFolders([folder]);
        setSelId(ps[0]?.id ?? null);
        setMainTab('gallery');
        setLoupeMode('grid');
      } catch (err: any) { notify(`❌ ${err.response?.data?.detail || "Failed to list photos"}`, "error"); }
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
      notify(`✅ Resumed — ${n} photos from ${savedFolders.length} folder${savedFolders.length !== 1 ? 's' : ''}`, 'success');
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
      notify(`✅ Added ${rawPhotos.length} photos from ${newFolder.split(/[\\/]/).pop()}`, 'success');
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
    } catch { notify("Could not open folder picker.", "error"); }
  }, [notify]);

  /* grade — uses SSE stream so large folders never time out */
  const handleGrade = useCallback(async (forceRescan = false, skipModal = false) => {
    const safePath = sanitizePath(folder);
    if (!safePath && folders.length === 0) { notify("Paste a valid folder path first.", "error"); return; }
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
          if (msg.desc)                   setGradeDesc(msg.desc);
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
            if (msg.mogco_error) notify(`⚠️ ${msg.mogco_error}`, 'error');
            // Transparency: report which grader actually ran, and warn (don't hide)
            // when Deep Grade was requested but silently fell back to Fast (SigLIP)
            // because there wasn't enough free RAM to load the vision model.
            const _graders = new Set<string>(ps.map((p: any) => p?.breakdown?._grader).filter(Boolean));
            const _usedDeep = _graders.has('qwen');
            setGraderUsed(scanMode ? 'scan' : _usedDeep ? 'deep' : 'fast');
            if (deepGrade && !scanMode && !_usedDeep) {
              notify('⚠️ Deep Grade fell back to Fast (SigLIP) — not enough free RAM to load the vision model. Close some apps and re-grade for full accuracy.', 'error');
            }
            notify(`✅ Graded ${msg.total} images${mogcoNote}${mogcoErr}`, 'success');
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
          notify(`⚠️ Grading stopped early — recovered ${n} graded photo${n !== 1 ? 's' : ''} from the last checkpoint.`, 'error');
        } else {
          notify(isStall ? '❌ No response from the grader — it may be stalled. Check the server log and retry.' : `❌ ${msg}`, 'error');
        }
      } catch {
        notify(`❌ ${msg}`, 'error');
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
    if (pool.length < 5) { notify(`Need 5+ graded images${filterNote} for a sequence`, 'error'); return; }
    setLoading(true);
    try {
      const res = await axios.post(`${API}/api/generate`, { photos: pool, seed: Math.floor(Math.random()*999999), avoid_paths: carousel.map((c: any) => c.path) });
      const d = res.data;
      setCarousel(Array.isArray(d) ? d : d.sequence);
      setSubjType(d.subject_type ?? null);
      setMainTab('gallery');
      notify('✅ Sequence generated', 'success');
    } catch (err: any) { notify(`❌ ${err.response?.data?.detail || "Failed"}`, "error"); }
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
      notify(`✅ Saved as "${name}"`, 'success');
    } catch (err: any) { notify(`❌ ${err.response?.data?.detail || "Failed"}`, 'error'); }
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
            const ok = outputs.filter((r: any) => r.success).length;
            if (ok === 0 && outputs.length === 0) {
              notify('Creative Direction ran but produced no outputs.', 'info');
            } else {
              notify(`✅ Creative Direction — ${ok}/${outputs.length} images styled`, 'success');
            }
            break outer;
          }
        }
      }
    } catch (err: any) {
      notify(`❌ Creative Direction failed: ${err.message || err}`, 'error');
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
        notify(`Sequence saved — ${data.count} images in ${data.story_dir.split(/[\\/]/).pop()}`, 'success');
        setUsedCount(data.used_total ?? 0);
      } else {
        notify(`Save failed: ${data.error}`, 'error');
      }
    } catch (err: any) {
      notify(`Save error: ${err.message}`, 'error');
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
      notify(`Clear failed: ${err.message}`, 'error');
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
    } catch (err: any) { notify(`❌ ${err.response?.data?.detail || `Failed to toggle ${type}`}`, 'error'); }
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
  const picks     = photos.filter(p => gl(p.grade) === 'Strong' && !redacted.has(p.path)).length;
  const mids      = photos.filter(p => gl(p.grade) === 'Mid'    && !redacted.has(p.path)).length;
  // Paths marked as used: server flags + photos committed to any saved sequence
  const allUsedPaths = useMemo(() =>
    new Set([...Array.from(used), ...saved.flatMap(s => s.sequence.map((p: any) => p.path))]),
  [used, saved]);
  const rejects   = photos.filter(p => gl(p.grade) === 'Weak'    && !redacted.has(p.path)).length;
  // Star counts within the current grade filter (for the filter bar labels)
  const gradeFiltered = filterGrade ? photos.filter(p => gl(p.grade) === filterGrade) : photos;
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
      <div style={{ position:'fixed', inset:0, background:'#0e0e13', display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:20 }}>
        {backendError ? (
          <>
            <span style={{ fontSize:28 }}>⚠️</span>
            <span style={{ fontSize:14, color:'#e05', letterSpacing:'.05em', textAlign:'center', maxWidth:340 }}>
              Could not connect to the backend.<br/>
              <span style={{ color:'#888', fontSize:12 }}>Make sure the app is running correctly and try again.</span>
            </span>
            <button onClick={() => { setBackendError(false); window.location.reload(); }}
              style={{ marginTop:8, padding:'6px 18px', borderRadius:6, border:'1px solid #333', background:'#1a1a22', color:'#aaa', cursor:'pointer', fontSize:13 }}>
              Retry
            </button>
          </>
        ) : (
          <>
            <div style={{ width:40, height:40, border:'3px solid #333', borderTopColor:'#7c6af7', borderRadius:'50%', animation:'spin .8s linear infinite' }}/>
            <span style={{ fontSize:14, color:'#888', letterSpacing:'.05em' }}>Starting FrameGrade…</span>
          </>
        )}
      </div>
    );
  }

  return (
    <div
      style={{ display:'flex', flexDirection:'column', height:'100vh', background:C.bg, overflow:'hidden',
        fontFamily:"'Helvetica Neue',-apple-system,BlinkMacSystemFont,system-ui,sans-serif", fontSize:15, color:C.text }}
      onDrop={handleDrop} onDragOver={e => { e.preventDefault(); e.stopPropagation(); }} onDragEnter={handleDragEnter} onDragLeave={handleDragLeave}
    >

      {/* Drag-and-drop overlay */}
      {dragOver && (
        <div style={{ position:'fixed', inset:8, zIndex:200, pointerEvents:'none', borderRadius:12,
          display:'flex', alignItems:'center', justifyContent:'center',
          background:'rgba(10,10,13,.88)', backdropFilter:'blur(6px)',
          border:`2px dashed ${C.accent}`,
        }}>
          <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:12 }}>
            <FolderOpen size={48} strokeWidth={1} style={{ color:C.accent }}/>
            <span style={{ fontSize:18, fontWeight:700, color:C.accent }}>Drop folder to load</span>
          </div>
        </div>
      )}

      {/* Toast */}
      {toast && (
        <div style={{
          position:'fixed', top:12, left:'50%', transform:'translateX(-50%)', zIndex:300,
          padding:'7px 16px', borderRadius:8, fontSize:13, fontWeight:500, whiteSpace:'nowrap',
          background: toast.type==='success' ? 'oklch(20% .1 148)' : toast.type==='error' ? 'oklch(18% .1 18)' : C.surf2,
          border:`1px solid ${toast.type==='success' ? 'oklch(48% .14 148)' : toast.type==='error' ? 'oklch(44% .14 18)' : C.bdr2}`,
          color:C.text, boxShadow:'0 8px 32px rgba(0,0,0,.7)', animation:'slideUp .3s cubic-bezier(.2,0,0,1)',
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
          "gemma3:4b":    "Spatial Judge",
          "qwen2.5vl:3b": "Vision Eye",
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
            fontSize: isOffline ? 13 : 12.5, fontWeight: isOffline ? 600 : 500,
            background: isOffline ? 'oklch(72% .19 55)' : 'oklch(18% .12 25)',
            borderBottom: isOffline ? '2px solid oklch(52% .22 50)' : '1px solid oklch(44% .18 25)',
            color: isOffline ? 'oklch(12% .04 55)' : 'oklch(85% .08 25)',
            display:'flex', alignItems:'center', gap:10, flexWrap:'wrap',
            boxShadow: isOffline ? '0 2px 8px oklch(0% 0 0 / .25)' : 'none',
          }}>
            <span style={{ fontSize: isOffline ? 16 : 14, flexShrink:0 }}>
              {isOffline ? '🔴' : '⚠'}
            </span>
            {isOffline ? (
              <span style={{ flex:1, minWidth:0 }}>
                <strong>Ollama is offline.</strong>{' '}
                Jury Critique, Creative Director, and photo annotations are unavailable.{' '}
                <span style={{ fontWeight:400, opacity:.85 }}>Start Ollama to enable AI features.</span>
              </span>
            ) : (
              <span style={{ flex:1, minWidth:0 }}>
                {missingOllama.length > 0 && <>Missing vision engines: <strong>{missingOllama.map(m => MODEL_DISPLAY[m] ?? m).join(", ")}</strong>{missingGguf.length > 0 ? " · " : ""}</>}
                {missingGguf.length > 0 && <>Missing local file{missingGguf.length > 1 ? "s" : ""}: <strong>{missingGguf.join(", ")}</strong> — place manually in <code style={{ fontSize:11 }}>models/</code></>}
              </span>
            )}
            {/* Ollama out-of-date — overrides all other controls */}
            {updateRequired ? (
              <>
                <span style={{ flex:1, minWidth:0, fontSize:12, fontWeight:600, color:'oklch(80% .15 25)' }}>
                  Your Ollama engine is out of date and cannot run these models.
                </span>
                <a
                  href="https://ollama.com/download"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ flexShrink:0, padding:'4px 12px', fontSize:12, fontWeight:700,
                    background:'oklch(44% .18 25)', color:'oklch(92% .05 25)',
                    border:'1px solid oklch(58% .2 25)', borderRadius:6, cursor:'pointer',
                    whiteSpace:'nowrap', textDecoration:'none' }}>
                  Download Ollama Update
                </a>
              </>
            ) : (
              <>
                {/* Generic error */}
                {downloadError && !isDownloading && (
                  <span style={{ fontSize:11.5, color:'oklch(72% .18 25)', fontWeight:600, flex:1, minWidth:0 }}>
                    ✕ {downloadError}
                  </span>
                )}
                {/* Download / Retry button */}
                {missingOllama.length > 0 && !isDownloading && (
                  <button
                    onClick={() => { setDownloadError(null); handleDownloadMissing(); }}
                    style={{ flexShrink:0, padding:'4px 12px', fontSize:12, fontWeight:700,
                      background: downloadError ? 'oklch(38% .18 25)' : 'oklch(44% .18 25)',
                      color:'oklch(92% .05 25)', border:`1px solid ${downloadError ? 'oklch(58% .2 25)' : 'oklch(55% .18 25)'}`,
                      borderRadius:6, cursor:'pointer', whiteSpace:'nowrap' }}>
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
                        padding:'2px 8px', borderRadius:4, fontSize:11, fontWeight:700, whiteSpace:'nowrap',
                        background: isGpu ? 'oklch(22% .09 145)' : isCpu ? 'oklch(22% .12 55)' : 'oklch(20% .04 0)',
                        border: `1px solid ${isGpu ? 'oklch(46% .14 145)' : isCpu ? 'oklch(52% .18 55)' : 'oklch(36% .04 0)'}`,
                        color: isGpu ? 'oklch(72% .16 145)' : isCpu ? 'oklch(80% .14 55)' : 'oklch(55% .04 0)',
                      }}>
                      {chip.display} {isGpu ? '✓ GPU' : isCpu ? '⚡ CPU' : '—'}
                    </span>
                  );
                })}
                {anyCpu && <span style={{ fontSize:11, color:'oklch(75% .12 55)', fontWeight:500 }}>VRAM pressure — inference may be slow</span>}
              </div>
            )}
            {/* Progress indicator while downloading */}
            {isDownloading && (
              <div style={{ flexShrink:0, display:'flex', alignItems:'center', gap:8 }}>
                <div style={{ width:120, height:6, background:'oklch(28% .08 25)', borderRadius:3, overflow:'hidden' }}>
                  <div style={{ width:`${downloadProgress}%`, height:'100%', background:'oklch(62% .18 145)', borderRadius:3, transition:'width .3s ease' }}/>
                </div>
                <span style={{ fontSize:11, whiteSpace:'nowrap', color:'oklch(75% .06 25)' }}>
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
                  color: isOffline ? 'oklch(20% .04 55)' : 'oklch(55% .06 25)', fontSize:16, lineHeight:1, padding:'2px 4px' }}>
                ✕
              </button>
            )}
          </div>
        );
      })()}

      {/* Pre-grade info modal */}
      {preGradeModal && (
        <div style={{ position:'fixed', inset:0, zIndex:400, background:'rgba(0,0,0,1)', display:'flex', alignItems:'center', justifyContent:'center' }}
          role="presentation"
          onClick={() => setPreGradeModal(null)}>
          <div ref={preGradeDialogRef} style={{ background:C.surf1, border:`1px solid ${C.bdr2}`, borderRadius:12, padding:'28px 32px', maxWidth:420, width:'90%', display:'flex', flexDirection:'column', gap:16 }}
            role="dialog" aria-modal="true" aria-labelledby="pregrade-title"
            onClick={e => e.stopPropagation()}>
            <div id="pregrade-title" style={{ fontSize:16, fontWeight:700, color:C.text }}>Before you start</div>

            {/* Vision Engine status */}
            <div style={{ display:'flex', flexDirection:'column', gap:8 }}>
              {!graderStatus?.draft_available ? (
                <div style={{ padding:'10px 14px', borderRadius:8, background:'oklch(22% .12 55 / .25)', border:'1px solid oklch(52% .18 55 / .4)' }}>
                  <div style={{ fontSize:13, fontWeight:700, color:'oklch(80% .14 55)', marginBottom:4 }}>
                    {graderStatus?.qwen_download_pct != null
                      ? `Downloading Vision Engine — ${graderStatus.qwen_download_pct}%`
                      : 'Vision Engine: downloading in background…'}
                  </div>
                  {graderStatus?.qwen_download_pct != null && (
                    <div style={{ height:4, background:'rgba(255,255,255,.1)', borderRadius:2, overflow:'hidden', marginBottom:8 }}>
                      <div style={{ height:'100%', width:`${graderStatus.qwen_download_pct}%`,
                        background:'oklch(70% .18 55)', borderRadius:2,
                        transition:'width .8s cubic-bezier(.2,0,0,1)' }}/>
                    </div>
                  )}
                  <div style={{ fontSize:12, color:C.text2, lineHeight:1.5 }}>
                    ~6 GB one-time download — runs automatically in the background.
                    {graderStatus?.qwen_download_pct != null
                      ? ' Grading will start automatically once complete.'
                      : ' You can start grading now; it will begin once the download finishes.'}
                  </div>
                </div>
              ) : graderStatus?.qwen_warm ? (
                <div style={{ padding:'10px 14px', borderRadius:8, background:'oklch(20% .09 145 / .3)', border:'1px solid oklch(46% .14 145 / .4)' }}>
                  <div style={{ fontSize:13, fontWeight:700, color:'oklch(72% .16 145)', marginBottom:4 }}>Vision Engine: warm and ready</div>
                  <div style={{ fontSize:12, color:C.text2 }}>Already loaded in VRAM. Grading will start immediately.</div>
                </div>
              ) : graderStatus?.qwen_loading ? (
                <div style={{ padding:'10px 14px', borderRadius:8, background:`${C.accent}18`, border:`1px solid ${C.accent}44` }}>
                  <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:6 }}>
                    <div style={{ width:10, height:10, borderRadius:'50%', border:`2px solid ${C.accent}`, borderTopColor:'transparent', animation:'spin .8s linear infinite', flexShrink:0 }}/>
                    <span style={{ fontSize:13, fontWeight:700, color:C.accent }}>Vision Engine: loading into VRAM…</span>
                  </div>
                  <div style={{ fontSize:12, color:C.text2 }}>
                    Loading model weights from disk. Takes <strong>~30–60 seconds</strong> — Start Culling will unlock automatically.
                  </div>
                </div>
              ) : (
                <div style={{ padding:'10px 14px', borderRadius:8, background:`${C.accent}18`, border:`1px solid ${C.accent}44` }}>
                  <div style={{ fontSize:13, fontWeight:700, color:C.accent, marginBottom:4 }}>Vision Engine: ready to load</div>
                  <div style={{ fontSize:12, color:C.text2 }}>
                    Model is cached on disk. Loading starts now — will be ready in <strong>~30–60 seconds</strong>.
                  </div>
                </div>
              )}

              {/* System-RAM readiness — is it clear to grade? (live, polled every 2 s) */}
              {(sysRam || graderStatus) && (() => {
                const r = ramReadiness(sysRam ?? graderStatus);
                if (r.level === 'unknown') return null;
                const card = {
                  clear:    { bg:'oklch(20% .09 145 / .3)', border:'oklch(46% .14 145 / .4)', color:'oklch(72% .16 145)', title:`✓ System memory: clear to grade`,    body:`${r.free?.toFixed(1)} GB free — plenty of headroom for a full cull.` },
                  tight:    { bg:'oklch(22% .12 55 / .25)',  border:'oklch(52% .18 55 / .4)',  color:'oklch(80% .14 55)',  title:`System memory: tight but OK`,       body:`${r.free?.toFixed(1)} GB free. Grading will run, but may drop to lighter CLIP scoring. Closing a few apps gives the best results.` },
                  critical: { bg:'oklch(24% .12 25 / .3)',   border:'oklch(52% .20 25 / .5)',  color:'oklch(80% .16 25)',  title:`Low system memory`,                body:`Only ${r.free?.toFixed(1)} GB free — below the ~${(sysRam?.ram_min_gb ?? graderStatus?.ram_min_gb ?? 1.8)} GB needed. Close some apps before grading or the cull may be refused.` },
                }[r.level]!;
                return (
                  <div style={{ padding:'10px 14px', borderRadius:8, background:card.bg, border:`1px solid ${card.border}` }}>
                    <div style={{ fontSize:13, fontWeight:700, color:card.color, marginBottom:4 }}>{card.title}</div>
                    <div style={{ fontSize:12, color:C.text2, lineHeight:1.5 }}>{card.body}</div>
                  </div>
                );
              })()}

              {/* One-time INT4 quantisation disclaimer — only until the
                  pre-quantised cache exists on disk */}
              {graderStatus?.draft_available && !graderStatus?.qwen_int4_cached && !graderStatus?.qwen_warm && (
                <div style={{ padding:'10px 14px', borderRadius:8, background:'oklch(22% .12 55 / .25)', border:'1px solid oklch(52% .18 55 / .4)' }}>
                  <div style={{ fontSize:13, fontWeight:700, color:'oklch(80% .14 55)', marginBottom:4 }}>
                    First cull: one-time engine optimisation
                  </div>
                  <div style={{ fontSize:12, color:C.text2, lineHeight:1.5 }}>
                    The Vision Engine will be compressed for your GPU on this run — expect a
                    pause of <strong>a few minutes around 52%</strong>. The result is saved,
                    so every cull after this one skips it and starts in seconds.
                  </div>
                </div>
              )}

              {/* Pipeline calibration warmup status */}
              {graderStatus?.warmup_running && (
                <div style={{ padding:'10px 14px', borderRadius:8, background:'oklch(20% .08 270 / .3)', border:'1px solid oklch(52% .14 270 / .35)' }}>
                  <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:4 }}>
                    <div style={{ width:9, height:9, borderRadius:'50%', border:'2px solid oklch(70% .16 270)', borderTopColor:'transparent', animation:'spin .8s linear infinite', flexShrink:0 }}/>
                    <span style={{ fontSize:13, fontWeight:700, color:'oklch(74% .14 270)' }}>Calibrating pipeline…</span>
                  </div>
                  <div style={{ fontSize:12, color:C.text2 }}>Running your best photos through the engine to warm up CUDA kernels. Start Culling will unlock when done.</div>
                </div>
              )}
              {graderStatus?.warmup_done && !graderStatus?.warmup_running && (
                <div style={{ display:'flex', alignItems:'center', gap:7, padding:'8px 12px', borderRadius:8,
                  background:'oklch(20% .09 145 / .2)', border:'1px solid oklch(46% .14 145 / .3)' }}>
                  <div style={{ width:7, height:7, borderRadius:'50%', background:'oklch(64% .18 145)', flexShrink:0 }}/>
                  <span style={{ fontSize:12, color:'oklch(70% .14 145)' }}>Pipeline calibrated — first cull of this session will be fast</span>
                </div>
              )}

              {/* Re-grade toggle */}
              <div style={{ display:'flex', gap:6 }}>
                {(['all','new'] as const).map(opt => {
                  const active = opt === 'all' ? rescanAll : !rescanAll;
                  return (
                    <button key={opt} onClick={() => setRescanAll(opt === 'all')}
                      style={{ flex:1, padding:'7px 10px', borderRadius:7, fontSize:12, fontWeight:600,
                        cursor:'pointer', border:`1px solid ${active ? C.accent : C.bdr2}`,
                        background: active ? `${C.accent}22` : 'transparent',
                        color: active ? C.accent : C.text2,
                        transition:'all .15s' }}>
                      {opt === 'all' ? 'Re-grade everything' : 'New photos only'}
                    </button>
                  );
                })}
              </div>
              <div style={{ fontSize:11, color:C.text3, paddingLeft:2, marginTop:-6 }}>
                {rescanAll
                  ? 'Every photo runs through the full pipeline.'
                  : 'Already-graded photos are skipped — only new additions are scored.'}
              </div>

              {/* Niche picker */}
              <div style={{ display:'flex', flexDirection:'column', gap:5 }}>
                <div style={{ display:'flex', alignItems:'center', gap:7, fontSize:11, fontWeight:600, color:C.text3, letterSpacing:'.06em', textTransform:'uppercase' }}>
                  Photography Niche
                  {nicheDetecting && (
                    <span style={{ display:'flex', alignItems:'center', gap:5, textTransform:'none', letterSpacing:0, fontWeight:500, color:C.text3 }}>
                      <div style={{ width:9, height:9, borderRadius:'50%', border:'2px solid currentColor', borderTopColor:'transparent', animation:'spin .8s linear infinite' }}/>
                      Detecting ideal niche…
                    </span>
                  )}
                  {!nicheDetecting && nicheRec?.detected && nicheRec?.preset === preset && (
                    <span style={{ textTransform:'none', letterSpacing:0, fontWeight:600, color:C.accent }}>
                      ✓ auto-selected
                    </span>
                  )}
                </div>
                <select
                  value={preset}
                  aria-label="Grading niche / preset"
                  onChange={e => setPreset(e.target.value)}
                  style={{ width:'100%', padding:'7px 10px', borderRadius:7, fontSize:13,
                    background:C.surf2, border:`1px solid ${C.bdr2}`, color:C.text,
                    cursor:'pointer', outline:'none' }}>
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
                <div style={{ fontSize:12, color:C.text3, paddingLeft:2 }}>
                  {preGradeModal.photoCount} photo{preGradeModal.photoCount !== 1 ? 's' : ''} in folder
                </div>
              )}
            </div>

            {/* Deep Grade toggle — default OFF = fast SigLIP zero-shot; ON = Qwen VLM */}
            <label style={{ display:'flex', alignItems:'flex-start', gap:10, cursor:'pointer',
              padding:'10px 12px', borderRadius:8, background:C.surf2, border:`1px solid ${C.bdr2}`, marginTop:4 }}>
              <input type="checkbox" checked={deepGrade} onChange={e => setDeepGrade(e.target.checked)}
                style={{ marginTop:2, cursor:'pointer', accentColor:C.accent }} />
              <div>
                <div style={{ fontSize:13, fontWeight:600, color:C.text }}>Deep Grade (Qwen VLM)</div>
                <div style={{ fontSize:11, color:C.text3, marginTop:2, lineHeight:1.4 }}>
                  Off: fast SigLIP zero-shot grading — light on GPU/RAM, recommended.
                  On: a vision-language model reads each photo (more nuanced, slower, heavier GPU use).
                </div>
              </div>
            </label>

            {/* Actions */}
            {(() => {
              const _notReady = graderStatus?.qwen_loading || graderStatus?.qwen_download_pct != null || graderStatus?.warmup_running;
              return (
                <div style={{ display:'flex', gap:10, justifyContent:'flex-end', marginTop:4 }}>
                  <button onClick={() => setPreGradeModal(null)}
                    style={{ padding:'7px 18px', borderRadius:7, fontSize:13, fontWeight:600, cursor:'pointer',
                      background:'transparent', border:`1px solid ${C.bdr2}`, color:C.text2 }}>
                    Cancel
                  </button>
                  <button
                    disabled={!!_notReady}
                    autoFocus
                    onClick={() => { setPreGradeModal(null); handleGrade(rescanAll, true); }}
                    style={{ padding:'7px 20px', borderRadius:7, fontSize:13, fontWeight:700,
                      cursor: _notReady ? 'not-allowed' : 'pointer',
                      background: _notReady ? C.surf3 : C.accent,
                      border:'none', color: _notReady ? C.text3 : '#fff',
                      display:'flex', alignItems:'center', gap:7,
                      transition:'background .2s, color .2s' }}>
                    {(graderStatus?.qwen_loading || graderStatus?.warmup_running) && (
                      <div style={{ width:10, height:10, borderRadius:'50%', border:'2px solid currentColor', borderTopColor:'transparent', animation:'spin .8s linear infinite' }}/>
                    )}
                    {graderStatus?.qwen_loading ? 'Loading Engine…' : graderStatus?.warmup_running ? 'Calibrating…' : 'Start Culling'}
                  </button>
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
      <header style={{ display:'flex', alignItems:'center', gap:8, padding:'0 14px', height:44, flexShrink:0, background:C.surf, borderBottom:`1px solid ${C.border}` }}>

        <button onClick={openBrowser}
          title="Open folder"
          style={{ display:'flex', alignItems:'center', gap:6, padding:'0 10px', height:30, borderRadius:7, fontSize:13, fontWeight:600, cursor:'pointer', flexShrink:0, background:'transparent', border:`1px solid ${C.bdr2}`, color:C.text3 }}>
          <FolderOpen size={13}/>
          {photos.length > 0 ? (folders.length > 1 ? `${folders.length} folders` : folder.split(/[\\/]/).pop()) : 'Open Folder'}
        </button>
        {photos.length > 0 && (
          <button onClick={openAddFolder}
            title="Add another folder"
            style={{ display:'flex', alignItems:'center', gap:5, padding:'0 10px', height:30, borderRadius:7, fontSize:13, fontWeight:600, cursor:'pointer', flexShrink:0, background:'transparent', border:`1px solid ${C.bdr2}`, color:C.text3 }}>
            <span style={{ fontSize:16, lineHeight:1 }}>+</span>
            Add Folder
          </button>
        )}

        <div style={{ flex:1 }}/>

        {/* Preset — hidden; value retained for grading logic */}

        {/* Vision Engine download progress chip */}
        {graderStatus?.qwen_download_pct != null && (
          <div style={{ display:'flex', alignItems:'center', gap:7, flexShrink:0, padding:'0 10px', height:26, borderRadius:5, fontSize:12, fontWeight:600, border:'1px solid oklch(52% .18 55 / .5)', color:'oklch(78% .15 55)', background:'oklch(18% .08 55 / .4)', overflow:'hidden', position:'relative' }}>
            {/* Animated fill */}
            <div style={{ position:'absolute', left:0, top:0, bottom:0, width:`${graderStatus.qwen_download_pct}%`, background:'oklch(52% .18 55 / .18)', transition:'width .8s cubic-bezier(.2,0,0,1)' }}/>
            <div style={{ width:6, height:6, borderRadius:'50%', background:'oklch(70% .18 55)', flexShrink:0, animation:'pulse 1.5s ease-in-out infinite', position:'relative' }}/>
            <span style={{ position:'relative' }}>Downloading {graderStatus.qwen_download_pct}%</span>
          </div>
        )}

        {/* Grader mode indicator */}
        {graderStatus && (() => {
          const m = graderStatus.last_mode;
          const isIqaHeads = m === 'iqa_heads';
          const isClip     = m === 'clip_only';
          const isIdle     = m === 'idle' || !m;
          const dot   = isIqaHeads ? '#22c55e' : isClip ? '#f59e0b' : C.text3;
          const label = isIqaHeads ? 'Deep Edit' : isClip ? 'Scout Mode' : 'Ready';
          const tip   = graderStatus.last_error ? `Error: ${graderStatus.last_error}` :
                        isIqaHeads ? 'Full vision pipeline — composition, light, and moment scored' :
                        isClip     ? 'Fast contact-sheet pass — style matching only' :
                        'No grading run yet';
          if (isIdle) return null;
          return (
            <div title={tip} style={{ display:'flex', alignItems:'center', gap:5, flexShrink:0, padding:'0 9px', height:26, borderRadius:5, fontSize:12, fontWeight:600, border:`1px solid ${C.bdr2}`, color:C.text3, background:C.surf2 }}>
              <div style={{ width:6, height:6, borderRadius:'50%', background:dot, flexShrink:0 }}/>
              {label}
            </div>
          );
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
          const label = isGpu ? 'GPU' : 'CPU';
          const chipBg     = isGpu ? 'oklch(18% .10 145 / .5)' : 'oklch(25% .10 55 / .5)';
          const chipBorder = isGpu ? 'oklch(46% .16 145 / .5)' : 'oklch(52% .18 55 / .4)';
          const chipColor  = isGpu ? 'oklch(74% .18 145)' : 'oklch(80% .16 55)';
          const dotColor   = isGpu ? '#22c55e' : '#f59e0b';
          return (
            <div title={tip} style={{ display:'flex', alignItems:'center', gap:5, flexShrink:0, padding:'0 9px', height:26, borderRadius:5, fontSize:12, fontWeight:700, border:`1px solid ${chipBorder}`, color:chipColor, background:chipBg }}>
              <div style={{ width:6, height:6, borderRadius:'50%', background:dotColor, flexShrink:0 }}/>
              {label}
              {vramStr && isGpu && (
                <span style={{ fontWeight:400, opacity:.75, fontSize:11 }}>{vramStr}</span>
              )}
            </div>
          );
        })()}

        {/* System RAM chip — live (polled every 2 s), tells the user whether it's clear to grade */}
        {(sysRam || graderStatus) && (() => {
          const r = ramReadiness(sysRam ?? graderStatus);
          if (r.level === 'unknown') return null;
          const palette = {
            clear:    { bg:'oklch(18% .10 145 / .5)', border:'oklch(46% .16 145 / .5)', color:'oklch(74% .18 145)', dot:'#22c55e' },
            tight:    { bg:'oklch(25% .10 55 / .5)',  border:'oklch(52% .18 55 / .4)',  color:'oklch(80% .16 55)',  dot:'#f59e0b' },
            critical: { bg:'oklch(25% .12 25 / .5)',  border:'oklch(52% .20 25 / .5)',  color:'oklch(78% .18 25)',  dot:'#ef4444' },
          }[r.level]!;
          return (
            <div title={r.tip} style={{ display:'flex', alignItems:'center', gap:5, flexShrink:0, padding:'0 9px', height:26, borderRadius:5, fontSize:12, fontWeight:700, border:`1px solid ${palette.border}`, color:palette.color, background:palette.bg }}>
              <div style={{ width:6, height:6, borderRadius:'50%', background:palette.dot, flexShrink:0 }}/>
              RAM
              <span style={{ fontWeight:400, opacity:.75, fontSize:11 }}>{r.readout}</span>
            </div>
          );
        })()}

        {/* Grade filter pills — only after grading */}
        {isDone && (
          <div style={{ display:'flex', alignItems:'center', gap:3, flexShrink:0, animation:'fadeIn .32s cubic-bezier(.2,0,0,1)' }}>
            {([['Strong', picks, C.strong] as const, ['Mid', mids, C.mid] as const, ['Weak', rejects, C.weak] as const]).map(([label, count, col]) => {
              const active = filterGrade === label;
              return (
                <button key={label}
                  onClick={() => setFilterGrade(active ? null : label)}
                  style={{ display:'flex', alignItems:'center', gap:5, padding:'0 9px', height:26, borderRadius:5, fontSize:13, fontWeight:600,
                    cursor:'pointer', border:'none', outline:'none',
                    background: active ? `${col}22` : 'transparent',
                    boxShadow: active ? `0 0 0 1px ${col}66` : `0 0 0 1px ${C.bdr2}`,
                    color: active ? col : C.text3,
                    transition:'all .22s cubic-bezier(.2,0,0,1)' }}>
                  <div style={{ width:6, height:6, borderRadius:'50%', background:col, flexShrink:0 }}/>
                  {label}
                  <span style={{ fontWeight:400, opacity:.7 }}>{count}</span>
                </button>
              );
            })}
          </div>
        )}

        {isDone && graderUsed && (
          <div
            title={graderUsed === 'deep'
              ? 'Deep Grade: the Qwen vision model read each photo (highest accuracy).'
              : graderUsed === 'scan'
              ? 'Scan pass: SigLIP zero-shot only, technical scoring skipped (fastest).'
              : 'Fast grade: SigLIP zero-shot + TOPIQ. Enable Deep Grade (and free RAM) for the vision model.'}
            aria-label={`Graded in ${graderUsed === 'deep' ? 'Deep' : graderUsed === 'scan' ? 'Scan' : 'Fast'} mode`}
            style={{ display:'flex', alignItems:'center', gap:5, padding:'0 9px', height:26, borderRadius:5,
              fontSize:11.5, fontWeight:700, flexShrink:0, cursor:'default',
              color: graderUsed === 'deep' ? C.accent : C.text3,
              boxShadow: `0 0 0 1px ${graderUsed === 'deep' ? `${C.accent}66` : C.bdr2}` }}>
            {graderUsed === 'deep' ? '◆ Deep' : graderUsed === 'scan' ? '⚡ Scan' : '⚡ Fast'}
          </div>
        )}

        {isDone && <div style={{ width:1, height:18, background:C.bdr2, flexShrink:0 }}/>}

        {/* Score sort button */}
        {isDone && (
          <button onClick={() => setSortScore(s => s === null ? 'desc' : s === 'desc' ? 'asc' : null)}
            title={sortScore === 'desc' ? 'Sorted: Strong → Weak' : sortScore === 'asc' ? 'Sorted: Weak → Strong' : 'Sort by score'}
            style={{ display:'flex', alignItems:'center', gap:4, padding:'0 9px', height:26, borderRadius:5, cursor:'pointer', fontSize:12, fontWeight:600, flexShrink:0, transition:'all .25s cubic-bezier(.2,0,0,1)',
              background: sortScore ? C.surf3 : 'transparent',
              border: `1px solid ${sortScore ? C.aBdr : C.bdr2}`,
              color: sortScore ? C.accent : C.text3 }}>
            {sortScore === 'desc' ? <ArrowDown size={11}/> : sortScore === 'asc' ? <ArrowUp size={11}/> : <ArrowUpDown size={11}/>}
            Score
          </button>
        )}

        {isDone && <div style={{ width:1, height:18, background:C.bdr2, flexShrink:0 }}/>}

        {/* Tab switcher: Gallery / Sequence / Duplicates / Director */}
        {(() => {
          const dupCount = redacted.size > 0
            ? redacted.size
            : photos.filter(p => p.cluster_id >= 0 && !(p.sim_flag||'').includes('Best')).length;
          const tabs: [string, string, React.ReactNode][] = [
            ...(isDone ? [
              ['gallery',    'Gallery',                                  <LayoutGrid size={11}/>],
              ['duplicates', dupCount > 0 ? `Duplicates (${dupCount})` : 'Duplicates', <ImageOff size={11}/>],
              ['creative', `Creative${creativeResults.length ? ` (${creativeResults.filter((r:any)=>r.success).length})` : ''}`, <Wand2 size={11}/>],
            ] as [string,string,React.ReactNode][] : []),
          ];
          return (
            <div style={{ display:'flex', background:C.bg, borderRadius:6, border:`1px solid ${C.bdr2}`, overflow:'hidden', flexShrink:0, animation:'fadeIn .32s cubic-bezier(.2,0,0,1)' }}>
              {tabs.map(([id, label, icon], ti) => (
                <button key={id} onClick={() => { setMainTab(id as "gallery"|"duplicates"|"creative"); if (id === 'gallery') setLoupeMode('loupe'); }}
                  style={{ display:'flex', alignItems:'center', gap:5, padding:'0 11px', height:30, cursor:'pointer',
                    fontWeight:600, fontSize:13,
                    background: mainTab === id ? C.surf3 : 'transparent',
                    color: mainTab === id ? C.text : C.text3,
                    borderRight: ti < tabs.length - 1 ? `1px solid ${C.bdr2}` : 'none',
                    border:'none', outline:'none', transition:'background .22s ease, color .22s ease',
                  }}>
                  {icon}{label}
                </button>
              ))}
            </div>
          );
        })()}

        {/* Loupe / Grid — only in gallery tab */}
        {isDone && mainTab === 'gallery' && (
          <div style={{ display:'flex', background:C.bg, borderRadius:6, border:`1px solid ${C.bdr2}`, overflow:'hidden', flexShrink:0 }}>
            {([['loupe', <RectangleHorizontal size={12}/>, 'E'] as const, ['grid', <LayoutGrid size={12}/>, 'G'] as const]).map(([m, icon, key]) => (
              <button key={m} title={`${m==='loupe'?'Loupe':'Grid'} (${key})`} onClick={() => setLoupeMode(m)}
                style={{ display:'flex', alignItems:'center', justifyContent:'center', width:32, height:30, cursor:'pointer',
                  background: loupeMode===m ? C.surf3 : 'transparent',
                  color: loupeMode===m ? C.text : C.text3,
                  borderRight: m==='loupe' ? `1px solid ${C.bdr2}` : 'none',
                  border:'none', transition:'all .22s cubic-bezier(.2,0,0,1)' }}>
                {icon}
              </button>
            ))}
          </div>
        )}

        {/* Export */}
        {isDone && (
          <button onClick={() => setExportModal(true)}
            style={{ display:'flex', alignItems:'center', gap:5, padding:'0 10px', height:30, borderRadius:7, fontSize:13, fontWeight:600, cursor:'pointer', flexShrink:0, background:C.aLow, border:`1px solid ${C.aBdr}`, color:C.accent }}>
            <Download size={11}/> Export
          </button>
        )}

        {/* Sort Files button — appears after grading */}
        {isDone && (
          <button
            onClick={async () => {
              try {
                const res = await axios.post(`${API}/api/manage/sort-files`, {
                  folder_path: folders[0] || folder,
                  gallery: photos,
                  copy: false,
                });
                notify(`✅ Sorted ${res.data.moved} files into Strong / Mid / Weak`, 'success');
              } catch (err: any) {
                notify(`❌ Sort failed: ${err?.response?.data?.detail ?? err.message}`, 'error');
              }
            }}
            style={{ display:'flex', alignItems:'center', gap:5, padding:'0 10px', height:30,
              borderRadius:7, fontSize:13, fontWeight:600, cursor:'pointer', flexShrink:0,
              background:C.surf2, border:`1px solid ${C.bdr2}`, color:C.text2 }}>
            <ArrowUpDown size={11}/> Sort Files
          </button>
        )}

        {/* Scan mode toggle */}
        {!isGrading && (
          <button
            onClick={() => setScanMode(v => !v)}
            title={scanMode
              ? 'Low-Latency Scan: 1.5B drafts all shots, 7B Architect reviews top 20% only. Click to switch to Full.'
              : 'Full: 7B Architect reviews any shot where draft confidence ≤ 0.85. Click to switch to Scan.'}
            style={{
              display:'flex', alignItems:'center', gap:5, padding:'0 10px', height:30,
              borderRadius:7, fontSize:12, fontWeight:600, cursor:'pointer', flexShrink:0,
              background: scanMode ? 'oklch(72% .18 65 / .15)' : C.surf2,
              border:`1px solid ${scanMode ? 'oklch(72% .18 65 / .45)' : C.bdr2}`,
              color: scanMode ? 'oklch(72% .18 65)' : C.text3,
              transition:'background .25s ease, border-color .25s ease, color .25s ease',
            }}>
            <Zap size={11} fill={scanMode ? 'currentColor' : 'none'}/>
            Scan
          </button>
        )}

        {/* Grade button */}
        {isGrading ? (
          <div style={{ display:'flex', alignItems:'center', gap:8, padding:'0 12px', height:30, borderRadius:7, background:C.surf2, border:`1px solid ${C.bdr2}`, flexShrink:0, minWidth:190 }}>
            {/* Percent bar */}
            <div style={{ flex:1, height:4, background:C.surf3, borderRadius:2, overflow:'hidden' }}>
              <div style={{ height:'100%', width:`${Math.max(2, gradeProgress * 100)}%`, background:`linear-gradient(90deg,${C.accent},oklch(70% .19 205))`, borderRadius:2, transition:'width .4s cubic-bezier(.2,0,0,1)' }}/>
            </div>
            <span style={{ fontSize:11, fontWeight:700, color:C.accent, fontVariantNumeric:'tabular-nums', flexShrink:0 }}>
              {Math.round(gradeProgress * 100)}%
            </span>
            {gradeEtaSecs !== null && gradeEtaSecs > 3 && (
              <span style={{ fontSize:11, color:C.text3, flexShrink:0, fontVariantNumeric:'tabular-nums' }}>
                ~{gradeEtaSecs >= 60 ? `${Math.floor(gradeEtaSecs / 60)}m ${gradeEtaSecs % 60}s` : `${gradeEtaSecs}s`}
              </span>
            )}
          </div>
        ) : (
          <button onClick={() => handleGrade(true, false)}
            title="Grade all images (force fresh scores)"
            style={{
              display:'flex', alignItems:'center', gap:6, padding:'0 14px', height:30,
              borderRadius:7, flexShrink:0, fontSize:13, fontWeight:700, cursor:'pointer',
              background: isDone ? C.surf2 : (scanMode ? 'oklch(72% .18 65)' : C.accent),
              border:`1px solid ${isDone ? C.bdr2 : 'transparent'}`,
              color: isDone ? C.text2 : '#fff',
              animation: !isDone ? 'pulse 2.8s ease-in-out infinite' : 'none',
            }}>
            {scanMode ? <Zap size={12} fill="currentColor"/> : <Sparkles size={12}/>}
            {isDone ? (scanMode ? 'Re-scan' : 'Re-grade') : (scanMode ? 'Scan' : 'Grade')}
          </button>
        )}
      </header>

      {/* Progress bar + slogan */}
      <div style={{ flexShrink:0 }}
        role={isGrading ? "progressbar" : undefined}
        aria-label={isGrading ? "Grading progress" : undefined}
        aria-valuenow={isGrading ? Math.round(gradeProgress * 100) : undefined}
        aria-valuemin={isGrading ? 0 : undefined}
        aria-valuemax={isGrading ? 100 : undefined}
        aria-valuetext={isGrading && gradeDesc ? gradeDesc : undefined}>
        <div style={{ height:2, background:C.border, overflow:'hidden', position:'relative' }}>
          {listLoading && (
            <div style={{ position:'absolute', top:0, height:'100%', background:`linear-gradient(90deg,transparent,${C.accent},transparent)`, animation:'sweep 1.2s ease-in-out infinite' }}/>
          )}
          {!listLoading && isGrading && (
            <div style={{ height:'100%', width:`${Math.max(4, gradeProgress * 100)}%`, background:`linear-gradient(90deg,${C.accent},oklch(70% .19 205))`, transition:'width .35s cubic-bezier(.2,0,0,1)' }}/>
          )}
          {!listLoading && !isGrading && isDone && (
            <div style={{ height:'100%', width:'100%', background:`linear-gradient(90deg,${C.accent},oklch(70% .19 205))` }}/>
          )}
        </div>
        {isGrading && gradeDesc && (() => {
          // Surface the live photo counter (e.g. "44/100") that the backend
          // sends in gradeDesc — toSlogan() rewrites the message into a slogan
          // and would otherwise drop it. Keyed by slogan so the slogan fades in
          // on change while the counter updates in place per photo.
          const _count = (gradeDesc.match(/\d+\s*\/\s*\d+/) || [])[0] || '';
          return (
            <div key={toSlogan(gradeDesc)} style={{ padding:'3px 14px 4px', fontSize:10.5, color:C.text3, fontStyle:'italic', borderBottom:`1px solid ${C.border}`, animation:'fadeIn .4s cubic-bezier(.2,0,0,1)', display:'flex', gap:8, alignItems:'baseline', justifyContent:'space-between' }}>
              <span>{toSlogan(gradeDesc)}</span>
              {_count && <span style={{ fontStyle:'normal', fontVariantNumeric:'tabular-nums', color:C.text2, fontWeight:700, flexShrink:0 }}>{_count}</span>}
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
            <div style={{ padding:'3px 14px 4px', fontSize:10.5, color:C.text3, borderBottom:`1px solid ${C.border}`, display:'flex', gap:7, alignItems:'center', animation:'fadeIn .4s cubic-bezier(.2,0,0,1)' }}>
              <div style={{ width:9, height:9, borderRadius:'50%', border:'2px solid currentColor', borderTopColor:'transparent', animation:'spin .8s linear infinite' }}/>
              <span>{warmMsg}</span>
            </div>
          );
        })()}
      </div>

      {/* ── Star filter bar ────────────────────────────────────── */}
      {mainTab === 'gallery' && isDone && (
        <div style={{ flexShrink:0, display:'flex', alignItems:'center', gap:10, padding:'0 14px', height:34, background:C.surf, borderBottom:`1px solid ${C.border}` }}>
          <span style={{ fontSize:11, fontWeight:700, color:C.text3, textTransform:'uppercase', letterSpacing:'.08em', flexShrink:0 }}>Rating</span>
          <div style={{ display:'flex', gap:3 }}>
            {[1,2,3,4,5].map(n => {
              const count = starCounts[n];
              const active = filterStars === n;
              return (
                <button key={n} onClick={() => setFilterStars(active ? null : n)}
                  style={{ display:'flex', alignItems:'center', gap:5, padding:'3px 9px', borderRadius:5, cursor:'pointer', transition:'all .25s cubic-bezier(.2,0,0,1)',
                    background: active ? 'oklch(70% .18 72 / .14)' : 'transparent',
                    border: `1px solid ${active ? 'oklch(70% .18 72 / .5)' : C.bdr2}` }}>
                  <div style={{ display:'flex', gap:1.5 }}>
                    {[1,2,3,4,5].map(s => (
                      <svg key={s} width="8" height="8" viewBox="0 0 24 24"
                        fill={s <= n ? 'oklch(70% .18 72)' : 'none'}
                        stroke={s <= n ? 'oklch(70% .18 72)' : C.text3} strokeWidth="2">
                        <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
                      </svg>
                    ))}
                  </div>
                  <span style={{ fontSize:11, fontWeight:700, color: active ? 'oklch(70% .18 72)' : C.text3, minWidth:10, textAlign:'center' }}>{count}</span>
                </button>
              );
            })}
          </div>
          <div style={{ width:1, height:14, background:C.bdr2, flexShrink:0 }}/>
          <button onClick={() => setFilterStars(filterStars === 0 ? null : 0)}
            style={{ display:'flex', alignItems:'center', gap:5, padding:'3px 9px', borderRadius:5, cursor:'pointer', transition:'all .25s cubic-bezier(.2,0,0,1)',
              background: filterStars === 0 ? `${C.surf3}` : 'transparent',
              border: `1px solid ${filterStars === 0 ? C.bdr2 : C.bdr2}`, color: filterStars === 0 ? C.text2 : C.text3, fontSize:11, fontWeight:600 }}>
            Unrated <span style={{ color:C.text3, marginLeft:2 }}>{starCounts[0]}</span>
          </button>
          {filterStars !== null && (
            <button onClick={() => setFilterStars(null)}
              style={{ fontSize:11, color:C.text3, padding:'2px 6px', borderRadius:4, border:`1px solid ${C.bdr2}`, background:C.surf2, cursor:'pointer', marginLeft:2 }}>
              ✕ Clear
            </button>
          )}
          {redacted.size > 0 && (
            <>
              <div style={{ width:1, height:14, background:C.bdr2, flexShrink:0 }}/>
              <button onClick={() => setShowDuplicates(v => !v)}
                title={showDuplicates ? 'Hide duplicate shots' : 'Show duplicate shots'}
                style={{ display:'flex', alignItems:'center', gap:5, padding:'3px 9px', borderRadius:5, cursor:'pointer', transition:'all .25s cubic-bezier(.2,0,0,1)',
                  background: showDuplicates ? 'oklch(58% .18 18 / .14)' : 'transparent',
                  border: `1px solid ${showDuplicates ? 'oklch(58% .18 18 / .45)' : C.bdr2}`,
                  color: showDuplicates ? 'oklch(58% .18 18)' : C.text3, fontSize:11, fontWeight:600 }}>
                <Copy size={10}/>
                Dupes <span style={{ marginLeft:2 }}>{redacted.size}</span>
              </button>
            </>
          )}
          <span style={{ marginLeft:'auto', fontSize:11, color:C.text3 }}>{filteredPhotos.length} shown</span>
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
            <div style={{ flex:1, background:'#060609', display:'flex', alignItems:'center', justifyContent:'center', overflow:'hidden', position:'relative', minHeight:0, minWidth:0 }}>
              {photos.length === 0 ? (
                <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:12 }}>
                  <button
                    onClick={openBrowser}
                    style={{
                      display:'flex', flexDirection:'column', alignItems:'center', gap:16,
                      padding:'48px 64px', borderRadius:16, cursor:'pointer', background:'transparent',
                      border:`2px dashed ${dragOver ? '#3b82f6' : C.border}`,
                      transition:'all .28s cubic-bezier(.2,0,0,1)', outline:'none',
                    }}>
                    <FolderOpen size={48} strokeWidth={1.25} style={{ color: dragOver ? '#3b82f6' : C.text3, transition:'color .28s ease' }}/>
                    <span style={{ fontSize:20, fontWeight:500, color: dragOver ? '#3b82f6' : C.text2, transition:'color .28s ease' }}>
                      Drop a folder of street photos here to start
                    </span>
                    <span style={{ fontSize:13, fontWeight:400, color: C.text3 }}>
                      50–100 photos recommended for your first run
                    </span>
                  </button>
                  {catalogBanner && (
                    <div style={{ display:'flex', alignItems:'center', gap:10, padding:'10px 18px', background:C.surf2, border:`1px solid ${C.bdr2}`, borderRadius:10 }}>
                      <span style={{ fontSize:13, color:C.text2 }}>Resume last session?</span>
                      <button onClick={handleResume} style={{ padding:'4px 14px', fontSize:13, fontWeight:600, background:C.accent, color:'#fff', border:'none', borderRadius:7, cursor:'pointer' }}>Resume</button>
                      <button onClick={() => { axios.post(`${API}/api/catalog/clear`); setCatalogBanner(false); }} style={{ padding:'4px 10px', fontSize:13, color:C.text3, background:'transparent', border:`1px solid ${C.bdr2}`, borderRadius:7, cursor:'pointer' }}>Discard</button>
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
                      outline: selectedIds.has(selId ?? '') ? `3px solid ${C.accent}` : 'none',
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
                        width:34, height:34, borderRadius:8,
                        display:'flex', alignItems:'center', justifyContent:'center',
                        background: showEyeOverlay ? C.accent : 'rgba(10,10,13,.72)',
                        border: `1px solid ${showEyeOverlay ? C.accent : 'rgba(255,255,255,.12)'}`,
                        backdropFilter:'blur(8px)',
                        cursor:'pointer',
                        transition:'background .2s ease, border-color .2s ease, box-shadow .2s ease',
                        boxShadow: showEyeOverlay ? `0 0 0 3px ${C.accent}33` : '0 2px 8px rgba(0,0,0,.5)',
                        color: showEyeOverlay ? '#fff' : 'rgba(255,255,255,.75)',
                      }}
                      onMouseEnter={e => { if (!showEyeOverlay) (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,.12)'; }}
                      onMouseLeave={e => { if (!showEyeOverlay) (e.currentTarget as HTMLButtonElement).style.background = 'rgba(10,10,13,.72)'; }}
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
                      <line x1="33.33%" y1="0%" x2="33.33%" y2="100%" stroke="rgba(255,255,255,0.55)" strokeWidth="1"/>
                      <line x1="66.66%" y1="0%" x2="66.66%" y2="100%" stroke="rgba(255,255,255,0.55)" strokeWidth="1"/>
                      <line x1="0%" y1="33.33%" x2="100%" y2="33.33%" stroke="rgba(255,255,255,0.55)" strokeWidth="1"/>
                      <line x1="0%" y1="66.66%" x2="100%" y2="66.66%" stroke="rgba(255,255,255,0.55)" strokeWidth="1"/>
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
                    const weakCol  = 'oklch(72% .18 50)';

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
                                fill="#ffffff" fontSize={nameFs} fontWeight="700"
                                fontFamily="'SF Mono',ui-monospace,monospace"
                                stroke="rgba(0,0,0,0.75)" strokeWidth={sw*1.8} paintOrder="stroke fill">
                                <tspan fill={C.strong} fontWeight="800">{'✓ '}</tspan>{name}
                              </text>
                              <line x1={edge} y1={ty + nameFs*0.20}
                                    x2={edge + uw} y2={ty + nameFs*0.20}
                                stroke={C.strong} strokeWidth={ulThick} strokeLinecap="round"/>
                              <text x={edge} y={ty + nameFs*0.20 + labelFs*1.5}
                                fill={C.strong} fontSize={labelFs} fontWeight="600"
                                fontFamily="'SF Mono',ui-monospace,monospace" opacity={0.75}
                                stroke="rgba(0,0,0,0.65)" strokeWidth={sw*1.4} paintOrder="stroke fill">
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
                                fill="#ffffff" fontSize={nameFs} fontWeight="700"
                                fontFamily="'SF Mono',ui-monospace,monospace"
                                stroke="rgba(0,0,0,0.75)" strokeWidth={sw*1.8} paintOrder="stroke fill">
                                {name}<tspan fill={weakCol} fontWeight="800">{' ↑'}</tspan>
                              </text>
                              <line x1={W - edge - uw} y1={ty + nameFs*0.20}
                                    x2={W - edge}       y2={ty + nameFs*0.20}
                                stroke={weakCol} strokeWidth={ulThick} strokeLinecap="round"/>
                              <text x={W - edge} y={ty + nameFs*0.20 + labelFs*1.5}
                                textAnchor="end"
                                fill={weakCol} fontSize={labelFs} fontWeight="600"
                                fontFamily="'SF Mono',ui-monospace,monospace" opacity={0.75}
                                stroke="rgba(0,0,0,0.65)" strokeWidth={sw*1.4} paintOrder="stroke fill">
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
                            const gc = C.mid;
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
                                fill="rgba(4,4,9,0.86)" stroke={col} strokeWidth={sw*0.5} rx={sw*1.4}/>
                              {/* Accent bar */}
                              <rect x={chipX} y={chipY} width={sw*1.2} height={chipH}
                                fill={col} rx={sw*0.6}/>
                              <text x={chipX+pad} y={chipY+pad*0.6+titleFs}
                                fill={col} fontSize={titleFs} fontWeight="800"
                                fontFamily="'SF Mono',ui-monospace,monospace">{title}</text>
                              {tip && (
                                <text x={chipX+pad} y={chipY+pad*0.6+titleFs+tipFs*1.4}
                                  fill="rgba(255,255,255,0.88)" fontSize={tipFs} fontWeight="500"
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
                                stroke="rgba(0,0,0,0.45)" strokeWidth={r * 0.18}/>
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
                        background:'rgba(8,8,12,.78)', border:'1px solid rgba(255,255,255,.12)',
                        borderRadius:9, backdropFilter:'blur(10px)', pointerEvents:'none' }}>
                        <div style={{ fontSize:10, fontWeight:700, letterSpacing:'.06em', color:'rgba(255,255,255,.55)', textTransform:'uppercase', marginBottom:1 }}>Critique map</div>
                        {_present.map(([t, l]) => {
                          const c = tierHeat(t);
                          return (
                            <div key={t} style={{ display:'flex', alignItems:'center', gap:7 }}>
                              <span style={{ width:11, height:11, borderRadius:'50%', flexShrink:0,
                                background:`radial-gradient(circle, ${c} 0%, ${c}55 70%, transparent 100%)`,
                                boxShadow:`0 0 6px ${c}` }}/>
                              <span style={{ fontSize:11, color:'rgba(255,255,255,.85)' }}>{l}</span>
                              <span style={{ fontSize:10, fontWeight:700, color:'rgba(255,255,255,.45)', marginLeft:'auto', paddingLeft:8 }}>{_counts[t]}</span>
                            </div>
                          );
                        })}
                      </div>
                    );
                  })()}
                  <button onClick={() => hasPrev && setSelId(filteredPhotos[selIdx-1].id)} disabled={!hasPrev}
                    style={{ position:'absolute', left:12, top:'50%', transform:'translateY(-50%)', width:34, height:34, borderRadius:'50%', display:'flex', alignItems:'center', justifyContent:'center', background:'rgba(0,0,0,.55)', backdropFilter:'blur(12px)', color:hasPrev?C.text:C.text3, opacity:hasPrev?1:0, border:'1px solid rgba(255,255,255,.07)', pointerEvents:hasPrev?'auto':'none', cursor:'pointer', fontSize:18 }}>‹</button>
                  <button onClick={() => hasNext && setSelId(filteredPhotos[selIdx+1].id)} disabled={!hasNext}
                    style={{ position:'absolute', right:12, top:'50%', transform:'translateY(-50%)', width:34, height:34, borderRadius:'50%', display:'flex', alignItems:'center', justifyContent:'center', background:'rgba(0,0,0,.55)', backdropFilter:'blur(12px)', color:hasNext?C.text:C.text3, opacity:hasNext?1:0, border:'1px solid rgba(255,255,255,.07)', pointerEvents:hasNext?'auto':'none', cursor:'pointer', fontSize:18 }}>›</button>
                  {/* Select toggle */}
                  {selId && (() => {
                    const isSel = selectedIds.has(selId);
                    return (
                      <button onClick={() => setSelectedIds(prev => { const next = new Set(prev); next.has(selId) ? next.delete(selId) : next.add(selId); return next; })}
                        style={{ position:'absolute', bottom:16, left:16, display:'flex', alignItems:'center', gap:6, padding:'6px 12px', borderRadius:20, cursor:'pointer', transition:'all .25s cubic-bezier(.2,0,0,1)', background:isSel ? C.accent : 'rgba(0,0,0,.6)', backdropFilter:'blur(12px)', border:`1px solid ${isSel ? C.accent : 'rgba(255,255,255,.12)'}`, color:'#fff', fontSize:12, fontWeight:700 }}>
                        <div style={{ width:14, height:14, borderRadius:3, flexShrink:0, background:isSel?'#fff':'transparent', border:`1.5px solid ${isSel?C.accent:'rgba(255,255,255,.6)'}`, display:'flex', alignItems:'center', justifyContent:'center' }}>
                          {isSel && <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20,6 9,17 4,12"/></svg>}
                        </div>
                        {isSel ? 'Selected' : 'Select'}
                      </button>
                    );
                  })()}
                  {/* Floating action bar */}
                  {selectedIds.size > 0 && (
                    <div style={{ position:'absolute', bottom:16, left:'50%', transform:'translateX(-50%) translateX(40px)', display:'flex', alignItems:'center', gap:10, background:C.surf, border:`1px solid ${C.bdr2}`, borderRadius:12, padding:'10px 18px', boxShadow:'0 8px 40px rgba(0,0,0,.7)', backdropFilter:'blur(12px)', zIndex:50, whiteSpace:'nowrap', animation:'slideUp .3s cubic-bezier(.2,0,0,1)' }}>
                      <span style={{ fontSize:14, fontWeight:700, color:C.text }}>{selectedIds.size} selected</span>
                      <div style={{ width:1, height:16, background:C.bdr2 }}/>
                      <button onClick={handleCreateFromSelection}
                        style={{ display:'flex', alignItems:'center', gap:6, padding:'6px 14px', borderRadius:8, background:C.accent, border:'none', color:'#fff', fontSize:13, fontWeight:700, cursor:'pointer' }}>
                        <Layers size={11}/> Start Sequence
                      </button>
                      <button onClick={handleGenerate}
                        style={{ display:'flex', alignItems:'center', gap:6, padding:'6px 14px', borderRadius:8, background:C.surf2, border:`1px solid ${C.bdr2}`, color:C.text2, fontSize:13, fontWeight:600, cursor:'pointer' }}>
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
              onMouseEnter={e => (e.currentTarget.style.background = 'oklch(64% .19 248 / .3)')}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            />
            )}

            {/* Right panel */}
            {photos.length > 0 && <div style={{ width:rightW, flexShrink:0, background:C.surf, borderLeft:`1px solid ${C.border}`, display:'flex', flexDirection:'column', overflow:'hidden' }}>

              {/* Thumbnail */}
              {sel && (
                <div style={{ flexShrink:0, position:'relative', aspectRatio:'3/2', background:C.bg, overflow:'hidden' }}>
                  <img key={sel.path} src={thumbUrl(sel.path)} alt=""
                    style={{ width:'100%', height:'100%', objectFit:'cover', display:'block', animation:'fadeIn .32s cubic-bezier(.2,0,0,1)' }}/>
                  {isGraded && (
                    <div style={{ position:'absolute', inset:0, background:'linear-gradient(to top,rgba(0,0,0,.85) 0%,transparent 55%)', display:'flex', alignItems:'flex-end', padding:'10px 12px' }}>
                      <div style={{ display:'flex', alignItems:'center', gap:6, background:'rgba(0,0,0,.6)', backdropFilter:'blur(8px)', borderRadius:6, padding:'6px 12px', border:`1px solid ${gc(sel.grade)}44` }}>
                        <div style={{ width:8, height:8, borderRadius:'50%', background:gc(sel.grade), flexShrink:0 }}/>
                        <span style={{ fontSize:15, fontWeight:700, color:C.text }}>{gl(sel.grade)}</span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Filename + copy + stars */}
              {sel && (
                <div style={{ flexShrink:0, padding:'10px 14px', borderBottom:`1px solid ${C.border}` }}>
                  <div style={{ display:'flex', alignItems:'center', gap:6, marginBottom:6 }}>
                    <span style={{ flex:1, fontSize:13, fontWeight:600, color:C.text, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                      {sel.path.split(/[\\/]/).pop()}
                    </span>
                    <button onClick={handleCopyPath} title="Copy path"
                      style={{ display:'flex', alignItems:'center', gap:4, padding:'4px 7px', borderRadius:5, background:copied ? C.sLow : C.surf2, border:`1px solid ${C.bdr2}`, color:copied ? C.strong : C.text3, fontSize:11, cursor:'pointer', transition:'all .25s cubic-bezier(.2,0,0,1)' }}>
                      <Copy size={10}/>{copied ? 'Copied!' : ''}
                    </button>
                  </div>
                  <StarRating stars={sel.stars ?? 0} onSet={n => handleSetStars(sel.id, n)}/>
                  {/* Grade display — read-only */}
                  {isDone && (
                    <div style={{ display:'flex', gap:4, marginTop:8 }}>
                      {(['Strong ✅','Mid ⚠️','Weak ❌'] as const).map(g => {
                        const _sc = sel.score ?? 0;
                        const derivedGrade = _sc >= 0.60 ? 'Strong ✅' : _sc >= 0.41 ? 'Mid ⚠️' : 'Weak ❌';
                        const isActive = derivedGrade === g;
                        const col = g.includes('Strong') ? C.strong : g.includes('Mid') ? C.mid : C.weak;
                        return (
                          <div key={g}
                            style={{ flex:1, padding:'3px 0', borderRadius:5, fontSize:11, fontWeight:700,
                              textAlign:'center', userSelect:'none', pointerEvents:'none',
                              background: isActive ? `${col}22` : 'transparent',
                              border: `1px solid ${isActive ? col : C.bdr2}`,
                              color: isActive ? col : C.text3 }}>
                            {gl(g)}
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
                          padding:'5px 10px', borderRadius:6, cursor:'pointer',
                          background:'oklch(64% .19 248 / .08)', border:`1px solid ${C.aBdr}`,
                          color:C.accent, fontSize:11, fontWeight:600, transition:'all .22s cubic-bezier(.2,0,0,1)' }}>
                        <Layers size={10} style={{ flexShrink:0 }}/>
                        Best of {count} similar shots — view duplicates
                      </button>
                    );
                  })()}
                </div>
              )}

              {/* Tabs */}
              {sel && (
                <div style={{ flexShrink:0, display:'flex', borderBottom:`1px solid ${C.border}` }}>
                  {(isDone
                    ? [['breakdown','Breakdown'],['analysis','Analysis'],['exif','EXIF']]
                    : [['exif','EXIF']]
                  ).map(([id, label], mapIdx, arr) => (
                    <>
                      <button key={id}
                        onClick={() => setInfoTab(id as any)}
                        style={{ flex:1, height:34, fontSize:11.5, fontWeight:600, cursor:'pointer', background:'none', border:'none', borderBottom:`2px solid ${infoTab===id ? C.accent : 'transparent'}`, color:infoTab===id ? C.accent : C.text3, transition:'all .25s cubic-bezier(.2,0,0,1)', letterSpacing:'.03em', marginBottom:-1 }}>
                        {label}
                      </button>
                    </>
                  ))}
                </div>
              )}

              {/* Panel body */}
              <div style={{ flex:1, overflowY:'auto', padding:14 }}>
                {infoTab === 'exif' && (
                  sel
                    ? <ExifBlock exif={sel.exif ?? {}}/>
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
                        setDeepCritique({ narrative_arc: 'Critique unavailable — is Ollama running?', geometry_composition: '' });
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
                            borderRadius:7, alignSelf:'flex-start', cursor:'pointer',
                            fontWeight:700, fontSize:11.5, letterSpacing:'.03em',
                            border:`1px solid ${isAuditModeActive ? C.accent : C.border}`,
                            background: isAuditModeActive ? `${C.accent}22` : C.surf2,
                            color: isAuditModeActive ? C.accent : C.text2,
                            transition:'all .2s cubic-bezier(.2,0,0,1)' }}>
                          {isAuditModeActive ? <EyeOff size={12}/> : <Eye size={12}/>}
                          {isAuditModeActive ? 'Hide Critique' : 'Vision Critique'}
                          {deepCritiqueLoading && (
                            <span style={{ width:8, height:8, borderRadius:'50%',
                              border:'1.5px solid currentColor', borderTopColor:'transparent',
                              animation:'spin .8s linear infinite', display:'inline-block' }}/>
                          )}
                          {!deepCritiqueLoading && isAuditModeActive && reasoningOverlayUrl && (
                            <span style={{ width:5, height:5, borderRadius:'50%',
                              background:C.strong, flexShrink:0 }}/>
                          )}
                        </button>
                        {/* VERIFIED badge */}
                        {sel.is_verified && (
                          <div style={{ display:'inline-flex', alignItems:'center', gap:5, padding:'4px 10px', borderRadius:5, background:'oklch(65% .17 148 / .14)', border:'1px solid oklch(65% .17 148 / .35)', alignSelf:'flex-start' }}>
                            <div style={{ width:6, height:6, borderRadius:'50%', background:C.strong }}/>
                            <span style={{ fontSize:11, fontWeight:700, letterSpacing:'.08em', color:C.strong }}>VERIFIED</span>
                          </div>
                        )}
                        {/* Tier label */}
                        {tierWord && (
                          <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                            <div style={{ width:10, height:10, borderRadius:'50%', background:gradeCol, flexShrink:0 }}/>
                            <span style={{ fontSize:20, fontWeight:800, letterSpacing:'.08em', color:gradeCol }}>{tierWord.toUpperCase()}</span>
                          </div>
                        )}
                        {/* Verdict */}
                        {verdict && (
                          <p style={{ fontSize:12.5, color:C.text2, lineHeight:1.7, margin:0, fontStyle:'italic' }}>{verdict}</p>
                        )}
                        {/* Per-aspect observations */}
                        {obsLines.length > 0 && (
                          <div style={{ display:'flex', flexDirection:'column', gap:1,
                            borderRadius:8, overflow:'hidden', border:`1px solid ${C.border}` }}>
                            {obsLines.map((line, idx) => {
                              const colon = line.indexOf(':');
                              const label = colon > 0 ? line.slice(0, colon).trim() : '';
                              const note  = colon > 0 ? line.slice(colon + 1).trim() : line;
                              const bdKey = label === 'Moment' ? 'Narrative'
                                          : label === 'Human'  ? 'Human/Culture'
                                          : label;
                              const v    = typeof bd[bdKey] === 'number' ? bd[bdKey] as number : null;
                              const vpct = v !== null ? Math.round(v * 100) : null;
                              const bc   = v === null ? C.accent
                                         : v >= 0.6  ? C.strong
                                         : v >= 0.41 ? '#f5a623' : C.weak;
                              const isLast = idx === obsLines.length - 1;
                              return (
                                <div key={idx} style={{ padding:'10px 13px',
                                  background: idx % 2 === 0 ? C.surf2 : C.bg,
                                  borderBottom: isLast ? 'none' : `1px solid ${C.border}` }}>
                                  <div style={{ display:'flex', justifyContent:'space-between',
                                    alignItems:'center', marginBottom: v !== null ? 5 : 0 }}>
                                    {label && (
                                      <span style={{ fontSize:10, fontWeight:700,
                                        letterSpacing:'.08em', color:bc }}>{label.toUpperCase()}</span>
                                    )}
                                    {vpct !== null && (
                                      <span style={{ fontSize:10, fontWeight:600, letterSpacing:'.05em', textTransform:'uppercase',
                                        color:bc }}>{vpct >= 60 ? 'Strong' : vpct >= 41 ? 'Mid' : 'Weak'}</span>
                                    )}
                                  </div>
                                  {v !== null && (
                                    <div style={{ height:2, background:C.bg, borderRadius:1,
                                      overflow:'hidden', marginBottom:6 }}>
                                      <div style={{ width:`${vpct}%`, height:'100%',
                                        background:bc, borderRadius:1,
                                        transition:'width .5s cubic-bezier(.2,0,0,1)' }}/>
                                    </div>
                                  )}
                                  <p style={{ fontSize:11.5, color:C.text2, margin:0, lineHeight:1.6 }}>{note}</p>
                                </div>
                              );
                            })}
                          </div>
                        )}
                        {/* Best / Weakest */}
                        {footer && (
                          <p style={{ fontSize:11, color:C.text3, margin:0, letterSpacing:'.02em' }}>{footer.trim()}</p>
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
                              paddingTop:10, borderTop:`1px solid ${C.border}` }}>
                              <div style={{ display:'flex', alignItems:'center' }}>
                                <span style={{ fontSize:10, fontWeight:700, letterSpacing:'.1em', color:C.accent }}>
                                  VISION CRITIQUE
                                </span>
                              </div>
                              {_dnarr && (
                                <div style={{ animation:'fadeIn .4s cubic-bezier(.2,0,0,1)' }}>
                                  <span style={{ fontSize:9.5, fontWeight:700, letterSpacing:'.08em', color:C.text3 }}>NARRATIVE</span>
                                  <p style={{ fontSize:11.5, color:C.text2, lineHeight:1.65, margin:'4px 0 0' }}>{_dnarr}</p>
                                </div>
                              )}
                              {_dgeo && (
                                <div style={{ animation:'fadeIn .4s cubic-bezier(.2,0,0,1)' }}>
                                  <span style={{ fontSize:9.5, fontWeight:700, letterSpacing:'.08em', color:C.text3 }}>GEOMETRY</span>
                                  <p style={{ fontSize:11.5, color:C.text2, lineHeight:1.65, margin:'4px 0 0' }}>{_dgeo}</p>
                                </div>
                              )}
                              {_qwenCritique && !_hasDeep && (
                                <p style={{ fontSize:12, color:C.text2, lineHeight:1.7, margin:0,
                                  fontStyle:'italic', padding:'8px 10px', background:C.surf2,
                                  borderRadius:6, border:`1px solid ${C.border}` }}>
                                  {_qwenCritique}
                                </p>
                              )}
                              {_vbboxes.length > 0 && (
                                <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
                                  <span style={{ fontSize:9.5, fontWeight:700, letterSpacing:'.08em', color:C.text3 }}>SPATIAL ANCHORS</span>
                                  {_vbboxes.map((b, bi) => {
                                    const guide  = regionGuide(b.label);
                                    const dotCol = tierColor(guide.tier);
                                    const coach  = b.justification || guide.tip;
                                    return (
                                      <div key={bi} style={{ display:'flex', gap:8, alignItems:'flex-start',
                                        padding:'6px 10px', background:C.surf2, borderRadius:6,
                                        border:`1px solid ${C.border}` }}>
                                        <div style={{ width:6, height:6, borderRadius:'50%',
                                          background:dotCol, flexShrink:0, marginTop:4 }}/>
                                        <div style={{ flex:1, minWidth:0 }}>
                                          <span style={{ fontSize:9.5, fontWeight:700, letterSpacing:'.06em',
                                            color:dotCol }}>{`${tierIcon(guide.tier)} ${guide.title}`.toUpperCase()}</span>
                                          {coach && (
                                            <p style={{ fontSize:11, color:C.text2, lineHeight:1.6, margin:'2px 0 0' }}>
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
                          <div style={{ fontSize:13, color:C.text2, lineHeight:1.75 }}>
                            {parseCritique(juryCritique, setCritTrigger, () => setCritTrigger(''))}
                          </div>
                        )}
                        {!rl && !juryCritique && (
                          <div style={{ display:'flex', flexDirection:'column', gap:8 }}>
                            <p style={{ fontSize:12, color:C.text3, lineHeight:1.7 }}>No grader analysis. Generate a jury critique:</p>
                            <button
                              onClick={() => sel && handleJuryCritique(sel.path)}
                              disabled={juryLoading}
                              style={{ display:'flex', alignItems:'center', gap:6, padding:'7px 14px', borderRadius:7, background:C.surf2, border:`1px solid ${C.bdr2}`, color:C.text2, fontSize:12, fontWeight:700, cursor: juryLoading ? 'wait' : 'pointer', alignSelf:'flex-start' }}>
                              {juryLoading
                                ? <><span style={{ width:10, height:10, borderRadius:'50%', border:`1.5px solid ${C.accent}`, borderTopColor:'transparent', animation:'spin .8s linear infinite', display:'inline-block' }}/> Generating…</>
                                : <><Wand2 size={11}/> Jury Critique</>}
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })() : (
                    <p style={{ fontSize:12, color:C.text3, lineHeight:1.7 }}>Grade your folder to see analysis.</p>
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
                      const ARCH_COLORS: Record<string,string> = {
                        geo: C.accent, night:'oklch(60% .19 280)', layer:'oklch(68% .18 148)',
                        messy:'oklch(68% .17 45)', maxdoc:'oklch(62% .18 0)'
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
                        s === 'good' ? C.strong : s === 'ok' ? '#f5a623' : s === 'bad' ? C.weak : C.text3;

                      // ── Telemetry Tags ─────────────────────────────────────────
                      const _ARCH_TAG: Record<string,{ icon:string; detail:string }> = {
                        geo:    { icon:'📐', detail:'Leading Lines / Symmetrical Geometry' },
                        layer:  { icon:'👥', detail:'Layered Depth — Foreground + Background' },
                        night:  { icon:'🌙', detail:'Night / Low-Key — Chiaroscuro' },
                        messy:  { icon:'⚡', detail:'Raw Street Energy' },
                        maxdoc: { icon:'📰', detail:'Dense Documentary Coverage' },
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
                        <div style={{ display:'flex', flexDirection:'column', gap:14, animation:'fadeIn .32s cubic-bezier(.2,0,0,1)' }}>

                          {/* ── Burst Context ─────────────────────────────────── */}
                          {_isBurst && (
                            <div style={{ padding:'10px 13px', borderRadius:8,
                              background: _isPrimary ? `${C.strong}0d` : `${C.mid}0d`,
                              border:`1px solid ${_isPrimary ? C.strong : C.mid}33` }}>
                              <div style={{ fontSize:9.5, fontWeight:700, letterSpacing:'.10em', color:C.text3, marginBottom:5 }}>
                                BURST SELECTION
                              </div>
                              <div style={{ display:'flex', alignItems:'baseline', gap:6, marginBottom:5 }}>
                                <span style={{ fontSize:11.5, fontWeight:800, letterSpacing:'.04em',
                                  color: _isPrimary ? C.strong : '#f5a623' }}>
                                  {_isPrimary ? '★ Primary Pick' : '↩ Alternate'}
                                </span>
                                {_isPrimary && _burstCnt > 0 && (
                                  <span style={{ fontSize:10.5, color:C.text3 }}>of {_burstCnt} similar frames</span>
                                )}
                              </div>
                              <p style={{ fontSize:11, color:C.text2, lineHeight:1.6, margin:0 }}>
                                {_isPrimary
                                  ? 'Highest-scoring frame in this burst. Alternate frames scored lower on overall quality.'
                                  : `${_altName || 'Another frame'} is the primary pick — scored higher. Compare before rejecting.`}
                              </p>
                            </div>
                          )}

                          {/* ── Grade verdict ─────────────────────────────────── */}
                          <div style={{ padding:'11px 13px', borderRadius:8,
                            background:`${_gradeColor}0f`, border:`1px solid ${_gradeColor}33` }}>
                            <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom: _gradeWhy ? 7 : 0 }}>
                              <div style={{ width:9, height:9, borderRadius:'50%', background:_gradeColor, flexShrink:0 }}/>
                              <span style={{ fontSize:13, fontWeight:800, letterSpacing:'.06em', color:_gradeColor }}>
                                {_tier.toUpperCase()}
                              </span>
                            </div>
                            {_gradeWhy && (
                              <p style={{ fontSize:12, color:C.text2, lineHeight:1.65, margin:0 }}>
                                {_gradeWhy}
                              </p>
                            )}
                          </div>

                          {/* ── Qwen one-liner ────────────────────────────────── */}
                          {qwenCritique && (
                            <p style={{ fontSize:11.5, color:C.text3, lineHeight:1.7, margin:0,
                              fontStyle:'italic', paddingLeft:10,
                              borderLeft:`2px solid ${C.border}` }}>
                              {qwenCritique}
                            </p>
                          )}

                          {/* ── Evidence Checklist ────────────────────────────── */}
                          {_checks.length > 0 && (
                            <div>
                              <div style={{ display:'flex', alignItems:'baseline', gap:8, marginBottom:8 }}>
                                <span style={{ fontSize:9.5, fontWeight:700, letterSpacing:'.09em',
                                  textTransform:'uppercase', color:C.text3 }}>Judge's Eye</span>
                                <span style={{ fontSize:9.5, color:C.text3, opacity:.6 }}>— what's working and what to fix</span>
                              </div>
                              <div style={{ display:'flex', flexDirection:'column',
                                borderRadius:7, overflow:'hidden', border:`1px solid ${C.border}` }}>
                                {_checks.map(({ key, label, value, state, isLimit, note }, ci) => {
                                  const col = _csCol(state);
                                  const showNote = !!note && (state === 'bad' || isLimit || _tier !== 'strong');
                                  return (
                                    <div key={key} style={{
                                      padding:'8px 11px',
                                      background: isLimit ? (
                                        _tier === 'weak' ? `${C.weak}14` : `${C.mid}14`
                                      ) : ci % 2 === 0 ? C.surf2 : C.bg,
                                      borderBottom: ci < _checks.length - 1 ? `1px solid ${C.border}` : 'none' }}>
                                      <div style={{ display:'flex', alignItems:'center', gap:10 }}>
                                        <div style={{ width:6, height:6, borderRadius:'50%',
                                          background:col, flexShrink:0 }}/>
                                        <span style={{ fontSize:10, fontWeight:700, letterSpacing:'.07em',
                                          color:C.text3, minWidth:62, textTransform:'uppercase' }}>{label}</span>
                                        <span style={{ fontSize:11.5, fontWeight:600, color:col }}>{value}</span>
                                        {isLimit && (
                                          <span style={{ marginLeft:'auto', fontSize:9, fontWeight:800,
                                            letterSpacing:'.08em', color: _tier === 'weak' ? C.weak : C.mid,
                                            textTransform:'uppercase' }}>
                                            {_tier === 'weak' ? '↑ WHAT FAILED' : '↑ WHAT TO FIX'}
                                          </span>
                                        )}
                                      </div>
                                      {showNote && (
                                        <p style={{ fontSize:10.5, color:C.text3, lineHeight:1.55,
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

                          {/* ── Telemetry Tags ────────────────────────────────── */}
                          {_teleTags.length > 0 && (
                            <div>
                              <div style={{ display:'flex', alignItems:'baseline', gap:8, marginBottom:8 }}>
                                <span style={{ fontSize:9.5, fontWeight:700, letterSpacing:'.09em',
                                  textTransform:'uppercase', color:C.text3 }}>Visual Language</span>
                                <span style={{ fontSize:9.5, color:C.text3, opacity:.6 }}>— the photographic tradition this frame is working in</span>
                              </div>
                              <div style={{ display:'flex', flexDirection:'column', gap:5, marginTop:8 }}>
                                {_teleTags.map(({ key, label, icon, detail, dominant }) => (
                                  <div key={key} style={{ display:'flex', alignItems:'center', gap:9,
                                    padding:'7px 11px', borderRadius:6,
                                    background: dominant ? `${C.accent}12` : C.surf2,
                                    border:`1px solid ${dominant ? C.accent + '40' : C.border}` }}>
                                    <span style={{ fontSize:13, lineHeight:1, flexShrink:0 }}>{icon}</span>
                                    <div style={{ flex:1, minWidth:0 }}>
                                      <span style={{ fontSize:10, fontWeight:700, letterSpacing:'.06em',
                                        color: dominant ? C.accent : C.text2,
                                        textTransform:'uppercase' }}>{label}</span>
                                      <span style={{ fontSize:10.5, color:C.text3, marginLeft:7 }}>{detail}</span>
                                    </div>
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
                      <Layers size={24} strokeWidth={1} style={{ color:C.text3 }}/>
                      <p style={{ fontSize:13, color:C.text3, textAlign:'center', lineHeight:1.6 }}>Grade your folder to see breakdown.</p>
                    </div>
                  )
                )}
              </div>

            </div>}

            </>)}
          </div>

          {/* ── Filmstrip (loupe mode only) ─────────────────────── */}
          {loupeMode === 'loupe' && photos.length > 0 && (
          <div style={{ flexShrink:0, background:C.surf, borderTop:`1px solid ${C.border}`, display:'flex', flexDirection:'column' }}>
            <div style={{ height:20, flexShrink:0, display:'flex', alignItems:'center', justifyContent:'space-between', padding:'0 12px', borderBottom:`1px solid ${C.border}` }}>
              <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                <span style={{ fontSize:10.5, color:C.text3, fontWeight:600, letterSpacing:'.08em', textTransform:'uppercase' }}>Library</span>
                {/* Tweaks toggle */}
                <button title="Filmstrip settings" onClick={() => setShowTweaks(v => !v)}
                  style={{ display:'flex', alignItems:'center', justifyContent:'center', width:18, height:16, cursor:'pointer', background:showTweaks ? C.surf3 : 'transparent', color:showTweaks ? C.accent : C.text3, border:'none', borderRadius:3, transition:'all .25s cubic-bezier(.2,0,0,1)' }}>
                  <SlidersHorizontal size={9}/>
                </button>
              </div>
              <span style={{ fontSize:10.5, color:C.text3, fontVariantNumeric:'tabular-nums', display:'flex', alignItems:'center', gap:5 }}>
                {isGrading && <span style={{ display:'inline-block', width:5, height:5, border:`1.5px solid ${C.accent}`, borderTopColor:'transparent', borderRadius:'50%', animation:'spin .8s linear infinite' }}/>}
                {isDone
                  ? <><span style={{ color:C.strong }}>{picks} picks</span>{'  ·  '}<span style={{ color:C.weak }}>{rejects} rejects</span>{'  ·  '}{photos.length} total</>
                  : `${photos.length} photos`}
              </span>
            </div>
            {/* Tweaks panel */}
            {showTweaks && (
              <div style={{ flexShrink:0, display:'flex', alignItems:'center', gap:16, padding:'6px 12px', borderBottom:`1px solid ${C.border}`, background:C.surf2 }}>
                <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                  <span style={{ fontSize:11, color:C.text3, whiteSpace:'nowrap' }}>Thumb size</span>
                  <input type="range" min={60} max={130} step={4} value={filmThumbH}
                    onChange={e => setFilmThumbH(Number(e.target.value))}
                    style={{ width:80, accentColor:C.accent, cursor:'pointer' }}/>
                  <span style={{ fontSize:11, color:C.text2, fontVariantNumeric:'tabular-nums', minWidth:22 }}>{filmThumbH}</span>
                </div>
                <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                  <span style={{ fontSize:11, color:C.text3 }}>Filenames</span>
                  <button onClick={() => setShowFilename(v => !v)}
                    style={{ position:'relative', width:28, height:16, borderRadius:8, border:'none', cursor:'pointer', padding:0, background:showFilename ? C.accent : C.bdr2, transition:'background .25s ease' }}>
                    <span style={{ position:'absolute', top:2, left:showFilename ? 13 : 2, width:12, height:12, borderRadius:'50%', background:'#fff', transition:'left .22s cubic-bezier(.2,0,0,1)', boxShadow:'0 1px 2px rgba(0,0,0,.3)' }}/>
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
            <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', background:C.bg, minHeight:0 }}>
              {/* Header */}
              <div style={{ flexShrink:0, display:'flex', alignItems:'center', gap:10, padding:'8px 16px', borderBottom:`1px solid ${C.border}`, background:C.surf }}>
                <span style={{ fontSize:14, fontWeight:700 }}>Similar Shots</span>
                <span style={{ fontSize:12, color:C.text3 }}>{groups.length} group{groups.length!==1?'s':''} · {totalDups} alternates</span>
                <div style={{ marginLeft:'auto' }}>
                  <button onClick={() => setExportModal(true)}
                    style={{ display:'flex', alignItems:'center', gap:5, padding:'0 10px', height:26, borderRadius:6, fontSize:12, fontWeight:600, cursor:'pointer', background:C.aLow, border:`1px solid ${C.aBdr}`, color:C.accent }}>
                    <Download size={11}/> Export
                  </button>
                </div>
              </div>

              {/* Flat grid — each group is a labeled section with auto-fill cells */}
              <div style={{ flex:1, overflowY:'auto', padding:12, minHeight:0 }}>
                {groups.map((g, gi) => {
                  const bestDc = gc(g.best.grade);
                  return (
                    <div key={gi} style={{ marginBottom: gi < groups.length - 1 ? 20 : 0 }}>

                      {/* Minimal group label row */}
                      <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:6 }}>
                        <div style={{ width:6, height:6, borderRadius:'50%', background:bestDc, flexShrink:0 }}/>
                        <span style={{ fontSize:11, fontWeight:700, color:C.text2 }}>
                          {g.all.length} similar shots
                        </span>
                        <span style={{ fontSize:10, color:C.text3 }}>
                          best <span style={{ color:C.text, fontWeight:600 }}>{Math.round(g.best.score * 100)}</span>
                        </span>
                        <div style={{ flex:1, height:1, background:C.border }}/>
                        <button
                          onClick={() => { setMainTab('gallery'); setSelId(g.best.id); setLoupeMode('loupe'); }}
                          style={{ fontSize:10, color:C.accent, padding:'2px 8px', borderRadius:4,
                            border:`1px solid ${C.aBdr}`, background:C.aLow, cursor:'pointer', fontWeight:600, flexShrink:0 }}>
                          Open best
                        </button>
                      </div>

                      {/* Auto-fill grid — same style as main gallery */}
                      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(150px, 1fr))', gap:5 }}>
                        {g.all.map((p: any, pi: number) => {
                          const isBest = pi === 0;
                          const dc     = gc(p.grade);
                          const delta  = isBest ? null : Math.round((p.score - g.best.score) * 100);
                          const fname  = (p.path.split(/[\\/]/).pop() ?? '').replace(/\.[^.]+$/, '');
                          return (
                            <button key={p.id}
                              onClick={() => { setMainTab('gallery'); setSelId(p.id); setLoupeMode('loupe'); }}
                              style={{ position:'relative', padding:0, border:'none', borderRadius:6,
                                overflow:'hidden', cursor:'pointer', display:'flex', flexDirection:'column',
                                background:C.surf,
                                outline: isBest ? `2px solid ${bestDc}` : `1px solid ${C.border}`,
                                outlineOffset: isBest ? -2 : -1,
                                transition:'outline .15s ease' }}>

                              {/* Image — cover fill, consistent 3:2 ratio */}
                              <div style={{ position:'relative', width:'100%', aspectRatio:'3/2', overflow:'hidden' }}>
                                <img src={thumbUrl(p.path)} alt="" loading="lazy"
                                  style={{ width:'100%', height:'100%', objectFit:'cover', display:'block',
                                    opacity: isBest ? 1 : 0.8 }}/>

                                {/* Gradient scrim */}
                                <div style={{ position:'absolute', inset:0, pointerEvents:'none',
                                  background:'linear-gradient(to bottom, rgba(0,0,0,.5) 0%, transparent 35%, transparent 55%, rgba(0,0,0,.55) 100%)' }}/>

                                {/* BEST / ALT badge — top left */}
                                <div style={{ position:'absolute', top:5, left:5, borderRadius:3,
                                  padding:'1px 5px', fontSize:8, fontWeight:800, letterSpacing:'.05em',
                                  background: isBest ? bestDc : 'rgba(0,0,0,.62)',
                                  color: isBest ? '#000' : 'rgba(255,255,255,.75)' }}>
                                  {isBest ? 'BEST' : 'ALT'}
                                </div>

                                {/* Score + delta — top right */}
                                <div style={{ position:'absolute', top:5, right:5, borderRadius:3,
                                  padding:'1px 5px', display:'flex', alignItems:'center', gap:3,
                                  background:'rgba(0,0,0,.62)', backdropFilter:'blur(4px)' }}>
                                  {delta !== null && (
                                    <span style={{ fontSize:8, fontWeight:700,
                                      color: delta < -10 ? '#f87171' : delta < 0 ? '#fbbf24' : '#86efac' }}>
                                      {delta > 0 ? '+' : ''}{delta}
                                    </span>
                                  )}
                                  <div style={{ width:4, height:4, borderRadius:'50%', background:dc }}/>
                                  <span style={{ fontSize:9, fontWeight:800, color:'#fff', fontVariantNumeric:'tabular-nums' }}>
                                    {Math.round(p.score * 100)}
                                  </span>
                                </div>
                              </div>

                              {/* Filename row — below image like gallery cells */}
                              <div style={{ padding:'3px 6px', background: isBest ? C.surf3 : C.surf }}>
                                <span style={{ fontSize:9.5, color: isBest ? C.text2 : C.text3,
                                  overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap',
                                  display:'block', fontFamily:"'SF Mono',monospace" }}>
                                  {fname}
                                </span>
                              </div>

                            </button>
                          );
                        })}
                      </div>

                    </div>
                  );
                })}

                {groups.length === 0 && (
                  <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', paddingTop:80, gap:10, color:C.text3 }}>
                    <ImageOff size={32} strokeWidth={1}/>
                    <p style={{ fontSize:14 }}>No duplicates detected.</p>
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
            Opener:   'oklch(60% .20 250)',
            Subject:  'oklch(65% .17 148)',
            Contrast: 'oklch(65% .20 55)',
            Detail:   'oklch(68% .16 90)',
            Closer:   'oklch(60% .20 290)',
          };
          const slotColor = (s: string) => SLOT_COLORS[s] ?? SLOT_COLORS[(s||'').charAt(0).toUpperCase()+(s||'').slice(1)] ?? C.text3;
          const ROLE_ORDER = ['Opener','Subject','Contrast','Detail','Closer','opener','subject','contrast','detail','closer'];
          const sortedPhotos = [...photos].sort((a,b) => {
            const r = (p:any) => gl(p.grade)==='Strong'?0:gl(p.grade)==='Mid'?1:2;
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
          <div style={{ flex:1, display:'flex', overflow:'hidden', background:C.bg }}>

            {/* ── Left config panel ───────────────────────────────── */}
            <div style={{ width:288, flexShrink:0, display:'flex', flexDirection:'column', borderRight:`1px solid ${C.border}`, background:C.surf, overflow:'hidden' }}>

              {/* Panel header */}
              <div style={{ padding:'14px 18px 12px', borderBottom:`1px solid ${C.border}`, flexShrink:0 }}>
                <div style={{ display:'flex', alignItems:'center', gap:7, marginBottom:4 }}>
                  <Wand2 size={14} style={{ color:C.accent }}/>
                  <span style={{ fontSize:14, fontWeight:700 }}>Creative Director</span>
                </div>
                <p style={{ fontSize:11, color:C.text3, lineHeight:1.5, margin:0 }}>
                  Curate 5 visually diverse shots into a cinematic story arc.
                </p>
              </div>

              {/* Scrollable config body */}
              <div style={{ flex:1, overflowY:'auto', padding:'18px 18px 8px', display:'flex', flexDirection:'column', gap:22 }}>

                {/* Brief */}
                <div>
                  <label style={{ display:'block', fontSize:11, fontWeight:700, letterSpacing:'.07em', textTransform:'uppercase', color:C.text2, marginBottom:8 }}>
                    Mood / Story Brief
                  </label>
                  <textarea
                    value={creativePrompt}
                    onChange={e=>setCreativePrompt(e.target.value)}
                    placeholder={`Describe the mood…\ne.g. "rainy evening, neon reflections"\nor "empty streets at dawn"`}
                    rows={4}
                    style={{ width:'100%', boxSizing:'border-box', resize:'none', background:C.bg, border:`1px solid ${C.bdr2}`, borderRadius:8, padding:'10px 12px', fontSize:12, color:C.text, lineHeight:1.6, outline:'none', fontFamily:'inherit' }}
                    onFocus={e=>{e.currentTarget.style.borderColor=C.aBdr}}
                    onBlur={e=>{e.currentTarget.style.borderColor=C.bdr2}}
                  />
                </div>

                {/* PDF Reference Library */}
                <div>
                  <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:6 }}>
                    <label style={{ fontSize:11, fontWeight:700, letterSpacing:'.07em', textTransform:'uppercase', color:C.text2 }}>
                      Photo Reference PDFs
                    </label>
                    {ragPdfs.length > 0 && (
                      <button onClick={handleRagClear}
                        style={{ fontSize:10, color:C.text3, background:'none', border:'none', cursor:'pointer', padding:0 }}>
                        clear all
                      </button>
                    )}
                  </div>
                  <p style={{ fontSize:11, color:C.text3, lineHeight:1.4, marginBottom:8 }}>
                    Upload photography books or reference PDFs. Concepts are extracted and blend into the grading anchor (30%).
                  </p>
                  {ragPdfs.length > 0 && (
                    <div style={{ marginBottom:8, display:'flex', flexDirection:'column', gap:4 }}>
                      {ragPdfs.map(p => (
                        <div key={p.name} style={{ display:'flex', alignItems:'center', gap:6, padding:'5px 8px', background:C.surf2, borderRadius:6, border:`1px solid ${C.bdr2}` }}>
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20,6 9,17 4,12"/></svg>
                          <span style={{ flex:1, fontSize:10, color:C.text2, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{p.name}</span>
                          <span style={{ fontSize:9, color:C.text3, flexShrink:0 }}>{p.phrases} phrases</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <label style={{ display:'flex', alignItems:'center', gap:7, padding:'8px 12px', border:`2px dashed ${ragUploading ? C.accent : C.bdr2}`, borderRadius:8, cursor: ragUploading ? 'wait' : 'pointer', color: ragUploading ? C.accent : C.text3, fontSize:12, transition:'border-color .2s, color .2s' }}>
                    {ragUploading
                      ? <span style={{ width:11, height:11, borderRadius:'50%', border:`1.5px solid ${C.accent}`, borderTopColor:'transparent', animation:'spin .7s linear infinite', display:'inline-block', flexShrink:0 }}/>
                      : <Upload size={13} strokeWidth={1.5}/>
                    }
                    <span>{ragUploading ? 'Extracting concepts…' : 'Add PDF reference'}</span>
                    <input type="file" accept="application/pdf" style={{ display:'none' }} disabled={ragUploading}
                      onChange={e => { const f = e.target.files?.[0]; if (f) handleRagUpload(f); e.target.value = ''; }}
                    />
                  </label>
                </div>

                {/* Peg reference upload */}
                <div>
                  <label style={{ display:'block', fontSize:11, fontWeight:700, letterSpacing:'.07em', textTransform:'uppercase', color:C.text2, marginBottom:4 }}>
                    Reference Peg <span style={{ fontWeight:400, textTransform:'none', letterSpacing:0, fontSize:10, color:C.text3 }}>optional · overrides anchor pool</span>
                  </label>
                  {pegFile ? (
                    <div style={{ display:'flex', alignItems:'center', gap:8, padding:'8px 10px', background:C.surf2, border:`1px solid ${pegHash ? C.accent : C.bdr2}`, borderRadius:8 }}>
                      {pegLoading
                        ? <span style={{ width:10, height:10, borderRadius:'50%', border:`1.5px solid ${C.accent}`, borderTopColor:'transparent', animation:'spin .7s linear infinite', display:'inline-block', flexShrink:0 }}/>
                        : <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke={pegHash ? C.accent : C.text3} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20,6 9,17 4,12"/></svg>
                      }
                      <span style={{ flex:1, fontSize:11, color:C.text2, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{pegFile.name}</span>
                      <button onClick={() => { setPegFile(null); setPegHash(null); }}
                        style={{ padding:0, border:'none', background:'transparent', color:C.text3, cursor:'pointer', fontSize:14, lineHeight:1 }}>✕</button>
                    </div>
                  ) : (
                    <label style={{ display:'flex', alignItems:'center', gap:7, padding:'8px 12px', border:`2px dashed ${C.bdr2}`, borderRadius:8, cursor:'pointer', color:C.text3, fontSize:12 }}>
                      <Upload size={13} strokeWidth={1.5}/>
                      <span>Upload reference image</span>
                      <input type="file" accept="image/*" style={{ display:'none' }}
                        onChange={e => { const f = e.target.files?.[0]; if (f) handlePegUpload(f); e.target.value = ''; }}
                      />
                    </label>
                  )}
                </div>

                {/* Sequence length */}
                <div>
                  <label style={{ display:'block', fontSize:11, fontWeight:700, letterSpacing:'.07em', textTransform:'uppercase', color:C.text2, marginBottom:8 }}>
                    Sequence Length
                  </label>
                  <select value={creativeCount} onChange={e => setCreativeCount(Number(e.target.value))}
                    style={{ width:'100%', height:36, borderRadius:7, fontSize:13, fontWeight:600, cursor:'pointer',
                      background:C.surf2, border:`1px solid ${C.bdr2}`, color:C.text2, padding:'0 10px',
                      appearance:'auto', outline:'none' }}>
                    {[3,4,5,6,7,8,9,10].map(n => (
                      <option key={n} value={n}>{n} photos</option>
                    ))}
                  </select>
                </div>

                {/* Reference photo */}
                <div>
                  <label style={{ display:'block', fontSize:11, fontWeight:700, letterSpacing:'.07em', textTransform:'uppercase', color:C.text2, marginBottom:4 }}>
                    Reference Photo <span style={{ fontWeight:400, textTransform:'none', letterSpacing:0, fontSize:10, color:C.text3 }}>optional</span>
                  </label>
                  <p style={{ fontSize:11, color:C.text3, lineHeight:1.4, marginBottom:10 }}>Sets the visual style anchor for the sequence.</p>
                  {creativeAnchor ? (
                    <div style={{ position:'relative', borderRadius:9, overflow:'hidden', border:`2px solid ${C.accent}`, cursor:'pointer', boxShadow:`0 0 0 3px ${C.accent}18` }}
                      onClick={()=>setCreativeAnchor(null)} title="Click to remove">
                      <img src={thumbUrl(creativeAnchor)} alt="" style={{ width:'100%', aspectRatio:'3/2', objectFit:'cover', display:'block' }}/>
                      <div style={{ position:'absolute', top:6, left:6, background:C.accent, borderRadius:4, padding:'2px 7px', fontSize:9, fontWeight:800, color:'#fff', letterSpacing:'.06em' }}>ANCHOR</div>
                      <div style={{ position:'absolute', top:6, right:6, background:'rgba(0,0,0,.65)', backdropFilter:'blur(4px)', borderRadius:5, padding:'3px 8px', fontSize:10, color:'rgba(255,255,255,.85)', fontWeight:600 }}>✕ remove</div>
                    </div>
                  ) : (
                    <div style={{ height:72, border:`2px dashed ${C.bdr2}`, borderRadius:9, display:'flex', alignItems:'center', justifyContent:'center', gap:7, color:C.text3, fontSize:12 }}>
                      <Wand2 size={14} strokeWidth={1.5}/>
                      <span>Click a photo below to set anchor</span>
                    </div>
                  )}
                </div>

                {/* Photo picker grid */}
                {sortedPhotos.length > 0 && (
                  <div>
                    <p style={{ fontSize:11, color:C.text3, marginBottom:8 }}>{sortedPhotos.length} photos · sorted by grade · click to anchor</p>
                    <div style={{ display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap:4 }}>
                      {sortedPhotos.map(p => {
                        const isAnchor = p.path===creativeAnchor;
                        const dc = gc(p.grade);
                        return (
                          <button key={p.id} onClick={()=>setCreativeAnchor(isAnchor?null:p.path)}
                            style={{ position:'relative', aspectRatio:'3/2', padding:0, border:'none', borderRadius:5, overflow:'hidden', cursor:'pointer',
                              outline: isAnchor?`2px solid ${C.accent}`:`1px solid ${dc}28`, outlineOffset:isAnchor?2:0,
                              transform:isAnchor?'scale(1.05)':'scale(1)', transition:'transform .12s, outline .12s' }}>
                            <img src={thumbUrl(p.path)} alt="" loading="eager" style={{ width:'100%', height:'100%', objectFit:'cover', display:'block' }}/>
                            <div style={{ position:'absolute', bottom:0, left:0, right:0, height:14, background:'linear-gradient(transparent, rgba(0,0,0,.75))', display:'flex', alignItems:'center', justifyContent:'flex-end', padding:'0 4px' }}>
                              <span style={{ fontSize:7, fontWeight:700, color:'#fff', fontVariantNumeric:'tabular-nums' }}>{Math.round(p.score*100)}</span>
                            </div>
                            {isAnchor && (
                              <div style={{ position:'absolute', inset:0, background:`${C.accent}40`, display:'flex', alignItems:'center', justifyContent:'center' }}>
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20,6 9,17 4,12"/></svg>
                              </div>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}

              </div>

              {/* Generate button — pinned to bottom */}
              <div style={{ padding:'14px 18px', borderTop:`1px solid ${C.border}`, flexShrink:0 }}>
                {/* Mode selector: Auto / Story / Competition */}
                <div style={{ display:'flex', gap:5, marginBottom:10 }}>
                  {(['auto','story','competition'] as const).map(m => (
                    <button key={m} onClick={()=>setSeqMode(m as any)}
                      style={{ flex:1, height:30, borderRadius:7, fontSize:11, fontWeight:700, cursor:'pointer', transition:'all .15s',
                        background: seqMode===m ? C.accent : C.surf2,
                        border: `1px solid ${seqMode===m ? C.accent : C.bdr2}`,
                        color: seqMode===m ? '#fff' : C.text2,
                        textTransform:'capitalize' }}>
                      {m}
                    </button>
                  ))}
                </div>
                {photos.length===0 && (
                  <p style={{ fontSize:11, color:C.text3, textAlign:'center', marginBottom:10 }}>Grade a folder first to load photos.</p>
                )}
                <button disabled={!canGenerate} onClick={handleRunCreativeDirection}
                  style={{ width:'100%', display:'flex', alignItems:'center', justifyContent:'center', gap:8, padding:'11px 0',
                    background: canGenerate ? C.accent : C.surf2, border:'none', borderRadius:8,
                    color: canGenerate ? '#fff' : C.text3, fontSize:14, fontWeight:700,
                    cursor: canGenerate ? 'pointer' : 'not-allowed', opacity:photos.length===0?0.45:1, transition:'all .18s' }}>
                  {creativeLoading
                    ? <><div style={{width:13,height:13,border:'2px solid #888',borderTopColor:'transparent',borderRadius:'50%',animation:'spin .8s linear infinite'}}/> Building sequence…</>
                    : <><Wand2 size={13}/> {hasResults ? 'Rebuild Sequence' : (
                        seqMode==='competition' ? 'Build Competition Set' :
                        seqMode==='auto'        ? 'Build Sequence (Auto)' :
                        'Build Story Sequence'
                      )}</>}
                </button>
                {usedCount>0 && (
                  <button onClick={handleClearUsed}
                    style={{ width:'100%', marginTop:6, fontSize:11, color:C.text3, background:'none', border:'none', cursor:'pointer', padding:'4px 0', textAlign:'center' }}>
                    Reset {usedCount} excluded photos
                  </button>
                )}
              </div>
            </div>

            {/* ── Right results panel ──────────────────────────────── */}
            <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden' }}>

              {/* Progress bar (only while loading) */}
              {creativeLoading && (
                <div style={{ flexShrink:0, padding:'12px 20px', borderBottom:`1px solid ${C.border}`, background:C.surf }}>
                  <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:8 }}>
                    <div style={{width:11,height:11,border:`2px solid ${C.accent}`,borderTopColor:'transparent',borderRadius:'50%',animation:'spin .8s linear infinite',flexShrink:0}}/>
                    <span style={{fontSize:13,color:C.text2,fontWeight:500}}>{creativeStage||'Building sequence…'}</span>
                    <span style={{marginLeft:'auto',fontSize:12,color:C.text3,fontVariantNumeric:'tabular-nums'}}>{Math.round(creativeProgress*100)}%</span>
                  </div>
                  <div style={{height:3,background:C.bdr2,borderRadius:2,overflow:'hidden'}}>
                    <div style={{height:'100%',width:`${Math.round(creativeProgress*100)}%`,background:`linear-gradient(90deg,${C.accent},oklch(70% .19 205))`,borderRadius:2,transition:'width .4s cubic-bezier(.2,0,0,1)'}}/>
                  </div>
                </div>
              )}

              {hasResults ? (
                <>
                  {/* Results toolbar */}
                  <div style={{flexShrink:0, display:'flex', alignItems:'center', justifyContent:'space-between', padding:'10px 20px', borderBottom:`1px solid ${C.border}`, background:C.surf}}>
                    <div style={{display:'flex', alignItems:'center', gap:10}}>
                      <span style={{fontSize:13, fontWeight:700}}>Story Sequence</span>
                      <span style={{fontSize:11, color:C.text3, background:C.surf2, borderRadius:4, padding:'2px 8px'}}>{successResults.length} images</span>
                      {creativeResults.some((r:any)=>!r.success) && (
                        <span style={{fontSize:11, color:C.weak, cursor:'default'}}
                          title={creativeResults.filter((r:any)=>!r.success).map((r:any)=>`${(r.source_path??'').split(/[\\/]/).pop()}: ${r.error??'failed'}`).join('\n')}>
                          {creativeResults.filter((r:any)=>!r.success).length} failed ⓘ
                        </span>
                      )}
                    </div>
                    <div style={{display:'flex', alignItems:'center', gap:8}}>
                      {!creativeLoading && (
                        <button disabled={sequenceSaving} onClick={handleSaveSequence}
                          style={{display:'flex', alignItems:'center', gap:5, fontSize:12, fontWeight:600, padding:'4px 12px', borderRadius:6,
                            cursor:sequenceSaving?'wait':'pointer', background:'transparent', border:`1px solid ${C.bdr2}`, color:C.text2, transition:'all .15s'}}>
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
                          <div key={i} style={{borderRadius:10, overflow:'hidden', border:`1px solid ${C.border}`, background:C.surf, display:'flex', flexDirection:'column', boxShadow:'0 2px 12px rgba(0,0,0,.25)'}}>
                            {/* Slot header */}
                            <div style={{padding:'8px 12px', background:C.surf2, borderBottom:`2px solid ${sc}`, display:'flex', alignItems:'center', gap:8}}>
                              <span style={{fontSize:9, fontWeight:800, letterSpacing:'.12em', color:sc, textTransform:'uppercase', flex:1}}>{slot}</span>
                              <span style={{fontSize:10, color:C.text3, fontWeight:600, background:C.surf3, borderRadius:3, padding:'1px 6px'}}>
                                {i+1}/{successResults.length}
                              </span>
                            </div>
                            {/* Photo — landscape 4:3 */}
                            <div style={{position:'relative', aspectRatio:'4/3', overflow:'hidden', background:C.bg}}>
                              <img src={photoUrl(r.source_path ?? r.output_path)} alt="" loading="eager" decoding="async"
                                style={{width:'100%', height:'100%', objectFit:'cover', display:'block'}}/>
                              <div style={{position:'absolute', inset:0, pointerEvents:'none',
                                background:'linear-gradient(to bottom, transparent 55%, rgba(0,0,0,.65) 100%)'}}/>
                              <a href={photoUrl(r.output_path ?? r.source_path)} download={fname} onClick={e=>e.stopPropagation()}
                                style={{position:'absolute', top:8, right:8, background:'rgba(0,0,0,.65)', backdropFilter:'blur(4px)', borderRadius:5, padding:'5px 8px', fontSize:10, color:'#fff', textDecoration:'none', display:'flex', alignItems:'center', gap:3, fontWeight:600, opacity:.85}}>
                                <Download size={9}/>
                              </a>
                              {photoScore!=null && (
                                <div style={{position:'absolute', bottom:8, right:10, display:'flex', alignItems:'center', gap:3,
                                  background:'rgba(0,0,0,.7)', backdropFilter:'blur(6px)', borderRadius:4, padding:'2px 8px'}}>
                                  <div style={{width:5, height:5, borderRadius:'50%', background:sc}}/>
                                  <span style={{fontSize:12, fontWeight:800, color:'#fff', fontVariantNumeric:'tabular-nums'}}>{Math.round(photoScore*100)}</span>
                                </div>
                              )}
                            </div>
                            {/* Filename */}
                            <div style={{padding:'8px 12px'}}>
                              <span style={{fontSize:10, color:C.text3, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', display:'block'}} title={fname}>{fname}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </>
              ) : (
                /* Empty state */
                <div style={{flex:1, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:18, color:C.text3, padding:40}}>
                  <Wand2 size={44} strokeWidth={1} style={{opacity:.3}}/>
                  <div style={{textAlign:'center', maxWidth:360}}>
                    <p style={{fontSize:16, fontWeight:700, color:C.text2, marginBottom:10}}>No sequence yet</p>
                    <p style={{fontSize:13, lineHeight:1.75, margin:0, color:C.text3}}>
                      Write a mood brief on the left,<br/>
                      optionally pick a reference photo,<br/>
                      then press <strong style={{color:C.accent, fontWeight:700}}>Build Story Sequence</strong>.
                    </p>
                    {photos.length===0 && (
                      <p style={{fontSize:12, color:C.weak, marginTop:14}}>Grade a folder first to load photos.</p>
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
      <div style={{ height:26, display:'flex', alignItems:'center', padding:'0 14px', gap:16, flexShrink:0, background:C.surf, borderTop:`1px solid ${C.border}` }}>
        <span style={{ fontSize:12, color:C.text2, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', flex:1, fontWeight:500 }}>
          {sel ? sel.path.split(/[\\/]/).pop() : 'Select a folder to begin'}
        </span>
        <div style={{ display:'flex', gap:12, flexShrink:0 }}>
          {[['← →','Navigate'],['H L','Navigate'],['1–5','Stars'],['G','Grid'],['E','Loupe']].map(([k, a]) => (
            <span key={k} style={{ fontSize:11, color:C.text3, display:'flex', alignItems:'center', gap:4 }}>
              <span style={{ background:C.surf2, border:`1px solid ${C.bdr2}`, borderRadius:3, padding:'1px 5px', fontSize:10.5, fontFamily:'monospace', color:C.text2 }}>{k}</span>{a}
            </span>
          ))}
        </div>
      </div>

      {/* ── Folder browser modal ────────────────────────────────── */}
      {showBrowser && (
        <div style={{ position:'fixed', inset:0, background:'rgba(0,0,0,.82)', backdropFilter:'blur(6px)', zIndex:50, display:'flex', alignItems:'center', justifyContent:'center', padding:16 }}>
          <div style={{ background:'#0f1218', border:'1px solid #1e242d', borderRadius:12, width:'100%', maxWidth:640, height:'82vh', display:'flex', flexDirection:'column', boxShadow:'0 24px 80px rgba(0,0,0,.8)' }}>
            <div style={{ flexShrink:0, display:'flex', alignItems:'center', justifyContent:'space-between', padding:'12px 20px', borderBottom:'1px solid #1e242d' }}>
              <span style={{ fontSize:15, fontWeight:600, color:'#fff' }}>Select Photo Folder</span>
              <button onClick={() => setShowBrowser(false)} style={{ color:'#50505e', cursor:'pointer', background:'none', border:'none' }}><X size={18}/></button>
            </div>
            <div style={{ flexShrink:0, display:'flex', alignItems:'center', gap:8, padding:'8px 16px', borderBottom:'1px solid #1e242d', background:'#0b0e14' }}>
              <button onClick={goUp} style={{ flexShrink:0, padding:'4px 10px', fontSize:13, color:'#9a9aaa', background:'#161b22', border:'1px solid #252d38', borderRadius:6, cursor:'pointer' }}>↑ Up</button>
              <span style={{ flex:1, fontSize:13, color:'#9a9aaa', fontFamily:'monospace', background:'#161b22', border:'1px solid #252d38', borderRadius:6, padding:'4px 10px', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{bPath}</span>
              <button
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
                style={{ flexShrink:0, padding:'4px 12px', fontSize:13, fontWeight:600, background:'#2563eb', color:'#fff', borderRadius:7, border:'none', cursor:bImages.length>0?'pointer':'not-allowed', opacity:bImages.length>0?1:0.4 }}>
                {browserMode === 'add' ? '+ Add' : 'Use Folder'}{bImages.length>0 ? ` (${bImages.length})` : ''}
              </button>
            </div>
            <div style={{ flex:1, display:'flex', overflow:'hidden' }}>
              <div style={{ width:140, flexShrink:0, borderRight:'1px solid #1e242d', padding:'10px 8px', display:'flex', flexDirection:'column', gap:2, background:'#0b0e14', overflowY:'auto' }}>
                <p style={{ fontSize:11, color:'#3a3a4a', textTransform:'uppercase', letterSpacing:'.08em', padding:'0 8px', marginBottom:6, fontWeight:600 }}>Quick access</p>
                {([
                  { label:'Desktop',   path:'C:\\Users\\Nicky Tuason\\Desktop' },
                  { label:'Pictures',  path:'C:\\Users\\Nicky Tuason\\Pictures' },
                  { label:'Downloads', path:'C:\\Users\\Nicky Tuason\\Downloads' },
                  { label:'Documents', path:'C:\\Users\\Nicky Tuason\\Documents' },
                  { label:'C:\\',      path:'C:\\' },
                ]).map(loc => (
                  <button key={loc.path} onClick={() => { setBPath(loc.path); loadBrowser(loc.path); }}
                    style={{ textAlign:'left', padding:'6px 10px', fontSize:13, borderRadius:7, color:bPath===loc.path?'#93c5fd':'#9a9aaa', background:bPath===loc.path?'rgba(37,99,235,.2)':'transparent', border:'none', cursor:'pointer', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                    {loc.label}
                  </button>
                ))}
              </div>
              <div style={{ flex:1, overflowY:'auto', padding:16 }}>
                {bLoading ? (
                  <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100%', flexDirection:'column', gap:10, color:'#3a3a4a' }}>
                    <div style={{ width:24, height:24, border:'2px solid #2563eb', borderTopColor:'transparent', borderRadius:'50%', animation:'spin .8s linear infinite' }}/>
                    <span style={{ fontSize:13 }}>Loading…</span>
                  </div>
                ) : bFolders.length===0 && bImages.length===0 ? (
                  <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height:'100%', color:'#3a3a4a', gap:8 }}>
                    <FolderOpen size={32} strokeWidth={1.5}/>
                    <p style={{ fontSize:14 }}>Empty folder</p>
                  </div>
                ) : (
                  <>
                    {bFolders.length > 0 && (
                      <div style={{ marginBottom:20 }}>
                        <p style={{ fontSize:11, color:'#3a3a4a', fontWeight:600, textTransform:'uppercase', letterSpacing:'.08em', marginBottom:8 }}>Folders ({bFolders.length})</p>
                        <div style={{ display:'grid', gap:6, gridTemplateColumns:'repeat(auto-fill, minmax(150px,1fr))' }}>
                          {bFolders.map((f, idx) => (
                            <button key={f} onClick={(e) => handleBrowserFolderClick(e as any, f, idx)}
                              style={{ display:'flex', alignItems:'center', gap:8, padding:'8px 12px', background: bSelFolders.has(f) ? 'rgba(37,99,235,.16)' : '#161b22', border: bSelFolders.has(f) ? '1px solid rgba(37,99,235,.4)' : '1px solid #252d38', borderRadius:8, cursor:'pointer', textAlign:'left' }}>
                              <FolderOpen size={13} style={{ color: bSelFolders.has(f) ? '#93c5fd' : '#60a5fa', flexShrink:0 }}/>
                              <span style={{ fontSize:13, color: bSelFolders.has(f) ? '#93c5fd' : '#c0c0d0', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{f.split(/[\\/]/).pop()}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {bImages.length > 0 && (
                      <div>
                        <p style={{ fontSize:11, color:'#3a3a4a', fontWeight:600, textTransform:'uppercase', letterSpacing:'.08em', marginBottom:8 }}>Images ({bImages.length})</p>
                        <div style={{ display:'grid', gap:6, gridTemplateColumns:'repeat(auto-fill, minmax(110px,1fr))' }}>
                          {bImages.slice(0,30).map(img => (
                            <div key={img} style={{ borderRadius:8, overflow:'hidden', border:'1px solid #1e242d', background:'#161b22' }}>
                              <img src={thumbUrl(img)} style={{ width:'100%', height:80, objectFit:'cover', display:'block' }} loading="lazy" alt=""/>
                              <p style={{ padding:'4px 6px', fontSize:11, color:'#50505e', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{img.split(/[\\/]/).pop()}</p>
                            </div>
                          ))}
                          {bImages.length > 30 && (
                            <div style={{ display:'flex', alignItems:'center', justifyContent:'center', borderRadius:8, border:'1px solid #1e242d', background:'#161b22', height:80 }}>
                              <span style={{ fontSize:13, color:'#50505e' }}>+{bImages.length-30} more</span>
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
