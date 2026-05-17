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
  Wand2, Zap,
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
          <div style={{ position:'absolute', bottom:3, left:3, width:6, height:6, borderRadius:'50%', background:gc(p.grade), boxShadow:`0 0 5px ${gc(p.grade)}99` }}/>
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
                      <span style={{ fontSize:12, fontWeight:800, color:'#fff', lineHeight:1, letterSpacing:'-.01em', fontVariantNumeric:'tabular-nums' }}>
                        {Math.round(p.score * 100)}
                      </span>
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

/* ── App ────────────────────────────────────────────────────────── */
export default function App() {
  const [folder,     setFolder]     = useState("");
  const [preset,     setPreset]     = useState("Classic Street");
  const [photos,     setPhotos]     = useState<any[]>([]);
  const [carousel,   setCarousel]   = useState<any[]>([]);
  const [saved,      setSaved]      = useState<{name: string; sequence: any[]}[]>([]);
  const [loading,      setLoading]      = useState(false);
  const [listLoading,  setListLoading]  = useState(false);
  const [gradeProgress, setGradeProgress] = useState(0);
  const [gradeDesc,     setGradeDesc]     = useState("");
  const [toast,      setToast]      = useState<{msg: string; type: "success"|"error"|"info"} | null>(null);
  const [selId,      setSelId]      = useState<string | null>(null);
  const [nicheRec,   setNicheRec]   = useState<any>(null);
  const [infoTab,    setInfoTab]    = useState<"exif"|"analysis">("analysis");
  const [scanMode,   setScanMode]   = useState(false);
  const [mainTab,    setMainTab]    = useState<"gallery"|"duplicates"|"creative">("gallery");
  const [seqMode,    setSeqMode]    = useState<'auto'|'director'>('auto');
  const [directorPrompt,  setDirectorPrompt]  = useState('');
  const [directorResult,  setDirectorResult]  = useState<any>(null);
  const [directorLoading, setDirectorLoading] = useState(false);
  const [directorPool,    setDirectorPool]    = useState<any[]>([]);
  const [mogcoTarget,     setMogcoTarget]     = useState(5);
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
  const [graderStatus,   setGraderStatus]   = useState<{last_mode:string,draft_available:boolean,verify_available:boolean,last_error:string|null}|null>(null);
  // ── Creative Direction state ──────────────────────────────────────────────
  const [creativeAnchor,   setCreativeAnchor]   = useState<string | null>(null);
  const [creativePrompt,   setCreativePrompt]   = useState("");
  const [creativeMode,     setCreativeMode]     = useState<"canny"|"depth">("canny");
  const [creativeCount,    setCreativeCount]    = useState(7);
  const [creativeLoading,  setCreativeLoading]  = useState(false);
  const [creativeProgress, setCreativeProgress] = useState(0);
  const [creativeStage,    setCreativeStage]    = useState("");
  const [creativeResults,     setCreativeResults]     = useState<any[]>([]);
  const [creativeOutDir,      setCreativeOutDir]      = useState("");
  const [creativeShowOriginal,setCreativeShowOriginal]= useState(false);
  const [usedCount,           setUsedCount]           = useState(0);
  const [sequenceSaving,      setSequenceSaving]      = useState(false);

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

  /* fetch grader model status on startup and after each grading run */
  const isDoneForStatus = !loading && photos.length > 0 && photos.some((p:any) => p.grade !== 'Pending');
  useEffect(() => {
    if (!backendReady) return;
    fetch(`${API}/api/models/status`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setGraderStatus(d); })
      .catch(() => {});
  }, [backendReady, isDoneForStatus]);

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
    axios.get(`${API}/api/exif`, { params: { path: sel.path } })
      .then(r => {
        if (Object.keys(r.data).length > 0)
          setPhotos(prev => prev.map(p => p.id === sel.id ? { ...p, exif: r.data } : p));
      })
      .catch(() => {});
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
      if (!showDuplicates && redacted.has(p.path)) return false;   // non-best duplicates hidden unless toggled
      const starsOk = filterStars === null || p.stars === filterStars;
      if (filterGrade) return gl(p.grade) === filterGrade && starsOk;
      if (carouselPaths.has(p.path)) return true;                   // sequence photos always visible when no grade filter
      return starsOk;
    });
    if (!sortScore) return base;
    return [...base].sort((a, b) => sortScore === 'desc' ? b.score - a.score : a.score - b.score);
  }, [photos, filterGrade, filterStars, redacted, showDuplicates, sortScore, carousel]);

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
    const load = async () => {
      setListLoading(true);
      try {
        const res = await axios.post(`${API}/api/list-folder`, { folder_path: sanitizePath(folder) });
        const rawPhotos: {path:string;exif:any}[] = res.data.photos || res.data.paths?.map((p: string) => ({path:p,exif:{}})) || [];
        if (!rawPhotos.length) notify("No images found in selected folder", "info");
        const ps = rawPhotos.map((p, i) => ({ id:`p-${i}`, path:p.path, grade:'Pending', score:0, breakdown:{}, critique:'', stars:0, exif:p.exif||{} }));
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
    axios.get(`${API}/api/catalog`)
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

  const handleResume = useCallback(async () => {
    try {
      const r = await axios.get(`${API}/api/catalog`);
      if (!r.data.exists || !r.data.photos?.length) return;
      const ps = r.data.photos.map((p: any, i: number) => ({ ...p, id: `p-${i}` }));
      const savedFolders: string[] = r.data.folders || [];
      // Apply same auto-redact logic as grading so duplicates are hidden in the gallery
      const autoRedacted = new Set<string>(
        ps.filter((p: any) => p.cluster_id >= 0 && !(p.sim_flag || '').startsWith('★'))
          .map((p: any) => p.path)
      );
      const firstVisible =
        ps.find((p: any) => !autoRedacted.has(p.path) && !((p.grade as string)?.includes('Weak'))) ??
        ps.find((p: any) => !autoRedacted.has(p.path));
      skipFolderLoadRef.current = true;
      setFolder(savedFolders[0] || '');
      setFolders(savedFolders);
      setPhotos(ps);
      setRedacted(autoRedacted);
      setSelId(firstVisible?.id ?? ps[0]?.id ?? null);
      setLoupeMode('grid');
      setCatalogBanner(false);
      notify(`✅ Resumed — ${ps.length} photos from ${savedFolders.length} folder${savedFolders.length !== 1 ? 's' : ''}`, 'success');
    } catch { notify('Failed to resume session', 'error'); }
  }, [notify]);

  const handleAddFolder = useCallback(async (newFolder: string) => {
    setListLoading(true);
    try {
      const res = await axios.post(`${API}/api/list-folder`, { folder_path: sanitizePath(newFolder) });
      const rawPhotos: {path:string;exif:any}[] = res.data.photos || [];
      setPhotos(prev => {
        const existing = new Set(prev.map(p => p.path));
        const added = rawPhotos
          .filter(p => !existing.has(p.path))
          .map((p, i) => ({ id:`p-${prev.length + i}`, path:p.path, grade:'Pending', score:0, breakdown:{}, critique:'', stars:0, exif:p.exif||{} }));
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
  const handleGrade = useCallback(async (forceRescan = false) => {
    const safePath = sanitizePath(folder);
    if (!safePath && folders.length === 0) { notify("Paste a valid folder path first.", "error"); return; }
    setLoading(true);
    setGradeProgress(0);
    setGradeDesc("");
    const allFolderPaths = folders.length > 0 ? folders.map(sanitizePath) : [safePath];
    try {
      const resp = await fetch(`${API}/api/grade/v2/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: allFolderPaths[0], folder_paths: allFolderPaths, preset, scan_mode: scanMode, force_rescan: forceRescan }),
      });
      if (!resp.ok) throw new Error(`Server error ${resp.status}`);
      const reader = resp.body!.getReader();
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
            setInfoTab('analysis');
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
            notify(`✅ Graded ${msg.total} images${mogcoNote}${mogcoErr}`, 'success');
            axios.post(`${API}/api/recommend`, { photos: msg.data })
              .then(rec => setNicheRec(rec.data))
              .catch(() => {});
            break outer;
          }
        }
      }
    } catch (err: any) { notify(`❌ ${err.message || 'Failed'}`, 'error'); }
    setLoading(false);
    setGradeProgress(0);
  }, [folder, folders, preset, notify]);

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
    if (!creativeAnchor) { notify('Select an anchor image first', 'error'); return; }
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
          anchor_path:    sanitizePath(creativeAnchor),
          folder_path:    sanitizePath(folders[0] || folder),
          style_prompt:   creativePrompt,
          structure_mode: creativeMode,
          n_target:       creativeCount,
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
  }, []);

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
      'Classic Street':       'The sequence carries the Magnum hallmarks: authentic gesture, layered framing, and a sense of life caught mid-breath.',
      'Travel Editor':         'The sequence reads like a dispatched edit — cultural immersion, sense of place, and subjects genuinely encountered rather than posed.',
      'Photojournalism':       'The sequence holds documentary weight: technically grounded, contextually honest, anchored in authentic human stakes.',
      'Cinematic/Editorial':   'Light is the connective tissue. The sequence moves through moods rather than subjects — each frame builds atmosphere for the next.',
      'Fine Art/Contemporary': 'The sequence operates conceptually — compositional logic over candid impulse, tonal control over spontaneous capture.',
      'Minimalist/Urbex':      'Structure drives the sequence. Negative space and geometric restraint create rhythm without relying on human narrative.',
      'LSPF (London Street)':  'The sequence has the quality of a slow walk through a city at dusk — atmospheric, human, unhurried.',
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
            <span style={{ fontSize:14, color:'#888', letterSpacing:'.05em' }}>Starting Street Story Curator…</span>
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

        {/* Grader mode indicator */}
        {graderStatus && (() => {
          const m = graderStatus.last_mode;
          const noModel = !graderStatus.draft_available;
          const isClip  = m === 'clip_only' || noModel;
          const isQwen  = m === 'qwen_fallback';
          const isSpec  = m === 'specvlm';
          const isIdle  = m === 'idle' || !m;
          const dot  = isClip ? '#ef4444' : isQwen ? '#f59e0b' : isSpec ? '#22c55e' : C.text3;
          const label= isClip ? 'CLIP only' : isQwen ? 'Qwen fallback' : isSpec ? (graderStatus.last_verify_used ? 'VLM + 7B' : 'VLM draft') : 'Ready';
          const tip  = graderStatus.last_error ? `Error: ${graderStatus.last_error}` :
                       isClip ? 'DeepSeek unavailable — grading with CLIP embeddings only' :
                       isQwen ? 'DeepSeek failed — using Qwen2.5-VL fallback' :
                       isSpec && graderStatus.last_verify_used ? 'DeepSeek 1.5B draft + 7B verification active' :
                       isSpec ? 'DeepSeek 1.5B draft-only (7B not available)' :
                       !graderStatus.draft_available ? 'DeepSeek weights not found' : 'No grading run yet';
          if (isIdle && graderStatus.draft_available) return null;
          return (
            <div title={tip} style={{ display:'flex', alignItems:'center', gap:5, flexShrink:0, padding:'0 9px', height:26, borderRadius:5, fontSize:12, fontWeight:600, border:`1px solid ${C.bdr2}`, color:C.text3, background:C.surf2 }}>
              <div style={{ width:6, height:6, borderRadius:'50%', background:dot, flexShrink:0 }}/>
              {label}
            </div>
          );
        })()}

        {/* Detected niche */}
        {nicheRec?.preset && (
          <div style={{ display:'flex', flexDirection:'column', justifyContent:'center', flexShrink:0, padding:'0 10px', height:30, borderRadius:6, background:C.surf2, border:`1px solid ${C.bdr2}`, animation:'fadeIn .32s cubic-bezier(.2,0,0,1)', lineHeight:1 }}>
            <span style={{ fontSize:9, fontWeight:700, letterSpacing:'.1em', textTransform:'uppercase', color:C.text3 }}>Detected niche</span>
            <span style={{ fontSize:13, fontWeight:600, color:C.text, marginTop:2 }}>{nicheRec.preset}</span>
          </div>
        )}

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
          const dupCount = photos.filter(p => p.cluster_id >= 0 && !(p.sim_flag||'').includes('Best')).length;
          const hasDups  = isDone && photos.some(p => p.cluster_id >= 0 && (p.sim_flag||'').includes('Best'));
          const tabs: [string, string, React.ReactNode][] = [
            ...(isDone ? [
              ['gallery',    'Gallery',                                  <LayoutGrid size={11}/>],
              ...(hasDups ? [['duplicates', `Duplicates (${dupCount})`, <ImageOff size={11}/>] as [string,string,React.ReactNode]] : []),
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
          <div style={{ display:'flex', alignItems:'center', gap:7, padding:'0 12px', height:30, borderRadius:7, background:C.surf2, border:`1px solid ${C.bdr2}`, color:C.text2, fontSize:12, fontWeight:600, flexShrink:0, minWidth:0 }}>
            <span style={{ width:10, height:10, borderRadius:'50%', border:`1.5px solid ${C.accent}`, borderTopColor:'transparent', animation:'spin .7s linear infinite', display:'inline-block', flexShrink:0 }}/>
            <span style={{ fontVariantNumeric:'tabular-nums', flexShrink:0 }}>{Math.round(gradeProgress * 100)}%</span>
            {gradeDesc && <span style={{ color:C.text3, fontSize:11, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', maxWidth:160 }}>{gradeDesc}</span>}
          </div>
        ) : (
          <button onClick={() => handleGrade(isDone)}
            title={isDone ? 'Re-grade all images (force full rescan)' : 'Grade new images only — already-graded images are skipped'}
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

      {/* Progress bar */}
      <div style={{ height:2, flexShrink:0, background:C.border, overflow:'hidden', position:'relative' }}>
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

            {loupeMode === 'grid' && (
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

            {loupeMode === 'loupe' && (<>

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
                      Select a folder or drag a folder here
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
                  <img
                    key={sel.path}
                    src={photoUrl(sel.path)}
                    alt=""
                    style={{ maxWidth:'100%', maxHeight:'100%', objectFit:'contain', display:'block', userSelect:'none', animation:'fadeIn .35s cubic-bezier(.2,0,0,1)', outline: selectedIds.has(selId ?? '') ? `3px solid ${C.accent}` : 'none', outlineOffset:'-3px', transition:'outline .22s ease' }}
                  />
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
                        <span style={{ fontSize:20, fontWeight:800, color:'#fff', fontVariantNumeric:'tabular-nums', fontFamily:'monospace', textShadow:'0 2px 8px rgba(0,0,0,.6)' }}>{Math.round(sel.score*100)}</span>
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
                        const isActive = sel.grade === g;
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
                    ? [['analysis','Analysis'],['exif','EXIF']]
                    : [['exif','EXIF']]
                  ).map(([id, label]) => (
                    <button key={id} onClick={() => setInfoTab(id as any)}
                      style={{ flex:1, height:34, fontSize:12.5, fontWeight:600, cursor:'pointer', background:'none', border:'none', borderBottom:`2px solid ${infoTab===id ? C.accent : 'transparent'}`, color:infoTab===id ? C.accent : C.text3, transition:'all .25s cubic-bezier(.2,0,0,1)', letterSpacing:'.03em', marginBottom:-1 }}>
                      {label}
                    </button>
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
                  isGrading ? (
                    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height:'100%', gap:8 }}>
                      <span style={{ width:20, height:20, borderRadius:'50%', border:`2px solid ${C.accent}`, borderTopColor:'transparent', animation:'spin .8s linear infinite', display:'inline-block' }}/>
                      <p style={{ fontSize:13, color:C.text3 }}>Analysing…</p>
                    </div>
                  ) : isGraded ? (
                    (() => {
                      const raw: Record<string,number> = sel?.breakdown ?? {};
                      // Strip meta-scores — only keep real CLIP aspect dimensions
                      // Fixed 5-dimension order — always shown uniformly on every photo
                      const ASPECT_KEYS = ['Technical','Composition','Lighting','Narrative','Human/Culture'] as const;
                      const META_KEYS = new Set(['aesthetic','personal','nima']);
                      // Merge raw breakdown; fall back to 0 so every bar is always present
                      const aspectMap: Record<string,number> = {};
                      Object.entries(raw).forEach(([k,v]) => {
                        if (!META_KEYS.has(k.toLowerCase())) aspectMap[k] = v as number;
                      });
                      // Build sorted list: known keys first in fixed order, then any extras
                      const known = ASPECT_KEYS.map(k => [k, aspectMap[k] ?? 0] as [string,number]);
                      const extra = Object.entries(aspectMap).filter(([k]) => !(ASPECT_KEYS as readonly string[]).includes(k));
                      const aspects = [...known, ...extra].sort((a,b) => b[1]-a[1]);
                      const gradeColor = gc(sel?.grade ?? '');
                      const pct  = Math.round((sel?.score ?? 0) * 100);
                      const tier = gl(sel?.grade ?? '');
                      const best    = aspects[0]?.[0] ?? '';
                      const weakest = aspects[aspects.length - 1]?.[0] ?? '';
                      return (
                        <div style={{ display:'flex', flexDirection:'column', gap:16, animation:'fadeIn .32s cubic-bezier(.2,0,0,1)' }}>
                          {/* Aspect bars */}
                          {aspects.length > 0 && (
                            <div style={{ display:'flex', flexDirection:'column', gap:9 }}>
                              {aspects.map(([k, v], idx) => {
                                const isTop = idx === 0;
                                const isBot = idx === aspects.length - 1;
                                const barCol = isTop ? C.strong : isBot ? C.weak : C.accent;
                                const labelCol = isTop ? C.strong : isBot ? C.weak : C.text2;
                                const vpct = Math.round(v * 100);
                                return (
                                  <div key={k}>
                                    <div style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline', marginBottom:4 }}>
                                      <span style={{ fontSize:11, fontWeight:600, color:labelCol, letterSpacing:'.01em' }}>{k}</span>
                                      <span style={{ fontSize:11, fontWeight:700, color:labelCol, fontVariantNumeric:'tabular-nums' }}>{vpct}</span>
                                    </div>
                                    <div style={{ height:3, background:C.surf3, borderRadius:2, overflow:'hidden' }}>
                                      <div style={{ height:'100%', width:`${vpct}%`, background:barCol, borderRadius:2, transition:'width .45s cubic-bezier(.2,0,0,1)' }}/>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          )}
                          {/* Best / weakest callout */}
                          {best && weakest && best !== weakest && (
                            <div style={{ display:'flex', gap:6 }}>
                              <span style={{ fontSize:11, color:C.strong, fontWeight:600, background:`${C.strong}18`, borderRadius:4, padding:'2px 7px' }}>↑ {best}</span>
                              <span style={{ fontSize:11, color:C.weak,   fontWeight:600, background:`${C.weak}18`,   borderRadius:4, padding:'2px 7px' }}>↓ {weakest}</span>
                            </div>
                          )}
                        </div>
                      );
                    })()
                  ) : (
                    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:8, padding:'20px 0' }}>
                      <Layers size={24} strokeWidth={1} style={{ color:C.text3 }}/>
                      <p style={{ fontSize:13, color:C.text3, textAlign:'center', lineHeight:1.6 }}>Grade your folder to see analysis.</p>
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
          const canGenerate = !creativeLoading && photos.length > 0;

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

                {/* Sequence length */}
                <div>
                  <label style={{ display:'block', fontSize:11, fontWeight:700, letterSpacing:'.07em', textTransform:'uppercase', color:C.text2, marginBottom:8 }}>
                    Sequence Length
                  </label>
                  <div style={{ display:'flex', gap:5 }}>
                    {[5,6,7,8].map(n => (
                      <button key={n} onClick={()=>setCreativeCount(n)}
                        style={{ flex:1, height:34, borderRadius:7, fontSize:13, fontWeight:700, cursor:'pointer', transition:'all .15s',
                          background:creativeCount===n?C.accent:C.surf2, border:`1px solid ${creativeCount===n?C.accent:C.bdr2}`,
                          color:creativeCount===n?'#fff':C.text2 }}>
                        {n}
                      </button>
                    ))}
                  </div>
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
                    : <><Wand2 size={13}/> {hasResults ? 'Rebuild Sequence' : 'Build Story Sequence'}</>}
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
