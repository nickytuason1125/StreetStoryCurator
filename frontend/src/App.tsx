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
} from "lucide-react";

const isTauri = () => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

const API = import.meta.env.VITE_API_URL
  || (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");
const thumbUrl = (p: string) => `${API}/api/thumb?path=${encodeURIComponent(p)}`;
const photoUrl = (p: string) => `${API}/api/photo?path=${encodeURIComponent(p)}`;

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
        <img src={thumbUrl(p.path)} alt="" decoding="async"
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
            style={{ transition:'fill .1s' }}>
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
  const rows: [string, string][] = [
    ['Camera',   exif.camera],
    ['Lens',     exif.lens],
    ['Focal',    exif.focal],
    ['Aperture', exif.aperture],
    ['Shutter',  exif.shutter],
    ['ISO',      exif.iso],
    ['Date',     exif.date],
    ['Time',     exif.time],
  ].filter(([, v]) => v) as [string, string][];
  return (
    <div style={{ display:'flex', flexDirection:'column', gap:0 }}>
      <p style={{ fontSize:11, fontWeight:700, letterSpacing:'.08em', textTransform:'uppercase', color:C.text3, marginBottom:8 }}>EXIF Data</p>
      {rows.map(([k, v]) => (
        <div key={k} style={{ display:'flex', justifyContent:'space-between', alignItems:'center', padding:'6px 0', borderBottom:`1px solid ${C.border}` }}>
          <span style={{ fontSize:12, color:C.text3, fontWeight:500 }}>{k}</span>
          <span style={{ fontSize:12, color:C.text, fontWeight:600, fontVariantNumeric:'tabular-nums', fontFamily:"'SF Mono',monospace" }}>{v}</span>
        </div>
      ))}
    </div>
  );
}

/* ── Export Modal ────────────────────────────────────────────────── */
function ExportModal({ photos, filterGrade, onClose }: { photos: any[]; filterGrade: string | null; onClose: () => void }) {
  const handleDownload = (p: any) => {
    const a = document.createElement('a');
    a.href = photoUrl(p.path); a.download = p.path.split(/[\\/]/).pop() || 'photo.jpg';
    a.click();
  };
  const handleDownloadAll = () => photos.forEach((p, i) => setTimeout(() => handleDownload(p), i * 200));
  return (
    <div style={{ position:'fixed', inset:0, zIndex:500, background:'rgba(0,0,0,.75)', backdropFilter:'blur(8px)', display:'flex', alignItems:'center', justifyContent:'center' }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div style={{ background:C.surf, border:`1px solid ${C.bdr2}`, borderRadius:12, width:560, maxHeight:'80vh', display:'flex', flexDirection:'column', boxShadow:'0 24px 80px rgba(0,0,0,.8)', overflow:'hidden', animation:'slideUp .22s cubic-bezier(.2,0,0,1)' }}>
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
            style={{ display:'flex', alignItems:'center', gap:5, padding:'4px 10px', borderRadius:6, fontSize:12, fontWeight:700, cursor:'pointer', background:selectMode ? C.aLow : 'transparent', border:`1px solid ${selectMode ? C.aBdr : C.bdr2}`, color:selectMode ? C.accent : C.text3, transition:'all .15s' }}>
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
                  outlineOffset:1, padding:0, border:'none', transition:'outline .1s',
                }}>
                <div style={{ position:'relative', width:'100%', aspectRatio:'3/2', background:C.surf2, overflow:'hidden' }}>
                  <img src={thumbUrl(p.path)} alt="" decoding="async" loading="lazy"
                    style={{ width:'100%', height:'100%', objectFit:'cover', display:'block', opacity: selectMode && !isChecked ? 0.55 : 1, transition:'opacity .15s' }}/>
                  {selectMode && (
                    <div style={{ position:'absolute', top:6, left:6, width:16, height:16, borderRadius:4, background:isChecked ? C.accent : 'rgba(0,0,0,.6)', border:`1.5px solid ${isChecked ? C.accent : 'rgba(255,255,255,.4)'}`, display:'flex', alignItems:'center', justifyContent:'center', transition:'all .15s' }}>
                      {isChecked && <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20,6 9,17 4,12"/></svg>}
                    </div>
                  )}
                  {isUsed && (
                    <div style={{ position:'absolute', top:5, right:5, background:'rgba(0,0,0,.75)', backdropFilter:'blur(4px)', borderRadius:3, padding:'1px 5px', display:'flex', alignItems:'center', gap:2 }}>
                      <Flag size={7} style={{ color:C.accent, flexShrink:0 }}/>
                      <span style={{ fontSize:9, fontWeight:700, color:C.accent }}>USED</span>
                    </div>
                  )}
                </div>
                <div style={{ padding:'4px 6px', background:isChecked ? `oklch(64% .19 248 / .1)` : isCurrent ? C.surf3 : C.surf, display:'flex', alignItems:'center', gap:4 }}>
                  {p.grade !== 'Pending' && gc(p.grade) !== C.text3 && (
                    <div style={{ width:6, height:6, borderRadius:'50%', background:gc(p.grade), flexShrink:0 }}/>
                  )}
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
        <div style={{ position:'absolute', bottom:16, left:'50%', transform:'translateX(-50%)', display:'flex', alignItems:'center', gap:10, background:C.surf, border:`1px solid ${C.bdr2}`, borderRadius:12, padding:'10px 18px', boxShadow:'0 8px 40px rgba(0,0,0,.7)', backdropFilter:'blur(12px)', zIndex:50, whiteSpace:'nowrap', animation:'slideUp .22s cubic-bezier(.2,0,0,1)' }}>
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
  const [preset,     setPreset]     = useState("Street - Magnum");
  const [photos,     setPhotos]     = useState<any[]>([]);
  const [carousel,   setCarousel]   = useState<any[]>([]);
  const [saved,      setSaved]      = useState<{name: string; sequence: any[]}[]>([]);
  const [loading,      setLoading]      = useState(false);
  const [listLoading,  setListLoading]  = useState(false);
  const [gradeProgress, setGradeProgress] = useState(0);
  const [toast,      setToast]      = useState<{msg: string; type: "success"|"error"|"info"} | null>(null);
  const [selId,      setSelId]      = useState<string | null>(null);
  const [nicheRec,   setNicheRec]   = useState<any>(null);
  const [infoTab,    setInfoTab]    = useState<"exif"|"analysis"|"breakdown">("exif");
  const [mainTab,    setMainTab]    = useState<"gallery"|"sequence"|"duplicates">("gallery");
  const [loupeMode,  setLoupeMode]  = useState<"loupe"|"grid">("loupe");
  const [subjType,   setSubjType]   = useState<string | null>(null);
  const [locked,     setLocked]     = useState<Set<string>>(new Set());
  const [used,       setUsed]       = useState<Set<string>>(new Set());
  const [redacted,   setRedacted]   = useState<Set<string>>(new Set());
  const [showBrowser,setShowBrowser]= useState(false);
  const [bPath,      setBPath]      = useState("C:\\Users");
  const [bFolders,   setBFolders]   = useState<string[]>([]);
  const [bImages,    setBImages]    = useState<string[]>([]);
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

  /* keyboard nav */
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (['INPUT','SELECT'].includes((document.activeElement as HTMLElement)?.tagName)) return;
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
  }, [photos, selId]);

  /* load photos when folder changes */
  useEffect(() => {
    if (!folder.trim()) return;
    const load = async () => {
      setListLoading(true);
      try {
        const res = await axios.post(`${API}/api/list-folder`, { folder_path: folder.trim() });
        const rawPhotos: {path:string;exif:any}[] = res.data.photos || res.data.paths?.map((p: string) => ({path:p,exif:{}})) || [];
        if (!rawPhotos.length) notify("No images found in selected folder", "info");
        const ps = rawPhotos.map((p, i) => ({ id:`p-${i}`, path:p.path, grade:'Pending', score:0, breakdown:{}, critique:'', stars:0, exif:p.exif||{} }));
        setPhotos(ps);
        setSelId(ps[0]?.id ?? null);
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

  const openBrowser = useCallback(() => { setShowBrowser(true); loadBrowser(bPath); }, [bPath, loadBrowser]);

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
  const handleGrade = useCallback(async () => {
    if (!folder.trim()) { notify("Paste a valid folder path first.", "error"); return; }
    setLoading(true);
    setGradeProgress(0);
    try {
      const resp = await fetch(`${API}/api/grade/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: folder.trim(), preset }),
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
          if (msg.error) throw new Error(msg.error);
          if (msg.done) {
            const ps = msg.data.map((p: any, i: number) => ({ ...p, id: `p-${i}` }));
            setPhotos(ps);
            setSelId(ps[0]?.id ?? null);
            setCarousel([]);
            setLoupeMode('loupe');
            setInfoTab('breakdown');
            notify(`✅ Graded ${msg.total} images`, 'success');
            const rec = await axios.post(`${API}/api/recommend`, { photos: msg.data });
            setNicheRec(rec.data);
            break outer;
          }
        }
      }
    } catch (err: any) { notify(`❌ ${err.message || 'Failed'}`, 'error'); }
    setLoading(false);
    setGradeProgress(0);
  }, [folder, preset, notify]);

  /* generate sequence */
  const handleGenerate = useCallback(async () => {
    const pool = photos
      .filter(p => p.grade !== 'Pending')
      .filter(p => seqMinStars === 0 || (p.stars ?? 0) >= seqMinStars);
    const filterNote = seqMinStars > 0 ? ` rated ${seqMinStars}★+` : '';
    if (pool.length < 5) { notify(`Need 5+ graded images${filterNote} for a sequence`, 'error'); return; }
    setLoading(true);
    try {
      const res = await axios.post(`${API}/api/generate`, { photos: pool, seed: Math.floor(Math.random()*999999) });
      const d = res.data;
      setCarousel(Array.isArray(d) ? d : d.sequence);
      setSubjType(d.subject_type ?? null);
      setMainTab('sequence');
      notify('✅ Sequence generated', 'success');
    } catch (err: any) { notify(`❌ ${err.response?.data?.detail || "Failed"}`, "error"); }
    setLoading(false);
  }, [photos, notify]);

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
  const picks     = photos.filter(p => p.grade.includes('Strong')).length;
  const mids      = photos.filter(p => p.grade.includes('Mid')).length;
  // Paths marked as used: server flags + photos committed to any saved sequence
  const allUsedPaths = useMemo(() =>
    new Set([...Array.from(used), ...saved.flatMap(s => s.sequence.map((p: any) => p.path))]),
  [used, saved]);
  const rejects   = photos.filter(p => p.grade.includes('Weak')).length;
  const filteredPhotos = useMemo(() => {
    const base = photos.filter(p => {
      const gradeOk = !filterGrade || p.grade.includes(filterGrade);
      const starsOk = filterStars === null || p.stars === filterStars;
      return gradeOk && starsOk && !redacted.has(p.path);
    });
    if (!sortScore) return base;
    return [...base].sort((a, b) => sortScore === 'desc' ? b.score - a.score : a.score - b.score);
  }, [photos, filterGrade, filterStars, redacted, sortScore]);
  // Star counts within the current grade filter (for the filter bar labels)
  const gradeFiltered = filterGrade ? photos.filter(p => p.grade.includes(filterGrade)) : photos;
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
      'Street - Magnum':       'The sequence carries the Magnum hallmarks: authentic gesture, layered framing, and a sense of life caught mid-breath.',
      'Travel Editor':         'The sequence reads like a dispatched edit — cultural immersion, sense of place, and subjects genuinely encountered rather than posed.',
      'World Press Doc':       'The sequence holds documentary weight: technically grounded, contextually honest, anchored in authentic human stakes.',
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
          color:C.text, boxShadow:'0 8px 32px rgba(0,0,0,.7)', animation:'slideUp .22s cubic-bezier(.2,0,0,1)',
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

        {/* Logo */}
        <div style={{ display:'flex', alignItems:'center', gap:7, flexShrink:0 }}>
          <div style={{ width:22, height:22, borderRadius:6, background:C.accent, display:'flex', alignItems:'center', justifyContent:'center', color:'#fff' }}>
            <Layers size={12}/>
          </div>
          <span style={{ fontSize:13, fontWeight:700, color:C.text2, letterSpacing:'.02em' }}>Street Story</span>
        </div>

        <div style={{ width:1, height:18, background:C.bdr2, flexShrink:0 }}/>

        {/* Folder path */}
        <div style={{ flex:1, display:'flex', alignItems:'center', gap:6, minWidth:0, background:C.bg, border:`1px solid ${C.bdr2}`, borderRadius:7, padding:'0 8px', height:30 }}>
          <FolderOpen size={12} style={{ color:C.text3, flexShrink:0 }}/>
          <input
            value={folder} onChange={e => setFolder(e.target.value)}
            placeholder="Paste folder path or browse…"
            onKeyDown={e => e.key === 'Enter' && handleGrade()}
            style={{ flex:1, background:'none', border:'none', outline:'none', color:C.text, fontSize:13, fontFamily:"'SF Mono','Fira Code',monospace", minWidth:0 }}
          />
          <button onClick={openBrowser}
            style={{ flexShrink:0, fontSize:13, fontWeight:500, padding:'0 8px', borderRadius:5, background:C.surf2, border:`1px solid ${C.bdr2}`, color:C.text2, cursor:'pointer', height:22 }}>
            Browse
          </button>
        </div>

        {/* Preset — hidden; value retained for grading logic */}

        {/* Detected niche */}
        {nicheRec?.preset && (
          <div style={{ display:'flex', flexDirection:'column', justifyContent:'center', flexShrink:0, padding:'0 10px', height:30, borderRadius:6, background:C.surf2, border:`1px solid ${C.bdr2}`, animation:'fadeIn .2s', lineHeight:1 }}>
            <span style={{ fontSize:9, fontWeight:700, letterSpacing:'.1em', textTransform:'uppercase', color:C.text3 }}>Detected niche</span>
            <span style={{ fontSize:13, fontWeight:600, color:C.text, marginTop:2 }}>{nicheRec.preset}</span>
          </div>
        )}

        {/* Grade filter pills — only after grading */}
        {isDone && (
          <div style={{ display:'flex', alignItems:'center', gap:3, flexShrink:0, animation:'fadeIn .2s' }}>
            {([['Strong', picks, C.strong] as const, ['Mid', mids, C.mid] as const, ['Weak', rejects, C.weak] as const]).map(([label, count, col]) => (
              <button key={label} onClick={() => {
                const next = filterGrade === label ? null : label;
                setFilterGrade(next);
                setLoupeMode('loupe');
                if (next) {
                  const first = photos.find(p => p.grade.includes(next));
                  if (first) setSelId(first.id);
                }
              }}
                style={{ display:'flex', alignItems:'center', gap:5, padding:'0 9px', height:26, borderRadius:5, cursor:'pointer', fontSize:13, fontWeight:600,
                  background: filterGrade === label ? `${col}22` : 'transparent',
                  border:`1px solid ${filterGrade === label ? `${col}66` : C.bdr2}`,
                  color: filterGrade === label ? col : C.text3,
                  transition:'all .15s' }}>
                <div style={{ width:6, height:6, borderRadius:'50%', background:col, flexShrink:0 }}/>
                {label}
                <span style={{ fontWeight:400, opacity:.7 }}>{count}</span>
              </button>
            ))}
            {filterGrade && (
              <button onClick={() => setFilterGrade(null)}
                style={{ fontSize:13, color:C.text3, padding:'0 7px', height:26, borderRadius:5, border:`1px solid ${C.bdr2}`, background:'transparent', cursor:'pointer' }}>
                ✕
              </button>
            )}
          </div>
        )}

        {isDone && <div style={{ width:1, height:18, background:C.bdr2, flexShrink:0 }}/>}

        {/* Score sort button */}
        {isDone && (
          <button onClick={() => setSortScore(s => s === null ? 'desc' : s === 'desc' ? 'asc' : null)}
            title={sortScore === 'desc' ? 'Sorted: Strong → Weak' : sortScore === 'asc' ? 'Sorted: Weak → Strong' : 'Sort by score'}
            style={{ display:'flex', alignItems:'center', gap:4, padding:'0 9px', height:26, borderRadius:5, cursor:'pointer', fontSize:12, fontWeight:600, flexShrink:0, transition:'all .15s',
              background: sortScore ? C.surf3 : 'transparent',
              border: `1px solid ${sortScore ? C.aBdr : C.bdr2}`,
              color: sortScore ? C.accent : C.text3 }}>
            {sortScore === 'desc' ? <ArrowDown size={11}/> : sortScore === 'asc' ? <ArrowUp size={11}/> : <ArrowUpDown size={11}/>}
            Score
          </button>
        )}

        {isDone && <div style={{ width:1, height:18, background:C.bdr2, flexShrink:0 }}/>}

        {/* Tab switcher: Gallery / Sequence / Duplicates */}
        {isDone && (() => {
          const dupCount = photos.filter(p => p.cluster_id >= 0 && !(p.sim_flag||'').includes('Best')).length;
          const hasDups  = photos.some(p => p.cluster_id >= 0 && (p.sim_flag||'').includes('Best'));
          const tabs: [string, string, React.ReactNode][] = [
            ['gallery',    'Gallery',                                  <LayoutGrid size={11}/>],
            ['sequence',   `Sequence${carousel.length ? ` (${carousel.length})` : ''}`, <Layers size={11}/>],
            ...(hasDups ? [['duplicates', `Duplicates (${dupCount})`, <ImageOff size={11}/>] as [string,string,React.ReactNode]] : []),
          ];
          return (
            <div style={{ display:'flex', background:C.bg, borderRadius:6, border:`1px solid ${C.bdr2}`, overflow:'hidden', flexShrink:0, animation:'fadeIn .2s' }}>
              {tabs.map(([id, label, icon]) => (
                <button key={id} onClick={() => { setMainTab(id as any); if (id === 'gallery') setLoupeMode('loupe'); }}
                  style={{ display:'flex', alignItems:'center', gap:5, padding:'0 11px', height:30, cursor:'pointer',
                    fontWeight:600, fontSize:13,
                    background: mainTab === id ? C.surf3 : 'transparent',
                    color: mainTab === id ? C.text : C.text3,
                    borderRight: id !== (hasDups ? 'duplicates' : 'sequence') ? `1px solid ${C.bdr2}` : 'none',
                    border:'none', outline:'none', transition:'background .12s, color .12s',
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
                  border:'none', transition:'all .12s' }}>
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

        {/* Grade button */}
        {isGrading ? (
          <div style={{ display:'flex', alignItems:'center', gap:7, padding:'0 14px', height:30, borderRadius:7, background:C.surf2, border:`1px solid ${C.bdr2}`, color:C.text2, fontSize:13, fontWeight:600, flexShrink:0 }}>
            <span style={{ width:10, height:10, borderRadius:'50%', border:`1.5px solid ${C.accent}`, borderTopColor:'transparent', animation:'spin .7s linear infinite', display:'inline-block', flexShrink:0 }}/>
            Grading…
          </div>
        ) : (
          <button onClick={handleGrade}
            style={{
              display:'flex', alignItems:'center', gap:6, padding:'0 14px', height:30,
              borderRadius:7, flexShrink:0, fontSize:13, fontWeight:700, cursor:'pointer',
              background: isDone ? C.surf2 : C.accent,
              border:`1px solid ${isDone ? C.bdr2 : 'transparent'}`,
              color: isDone ? C.text2 : '#fff',
              animation: !isDone ? 'pulse 2.8s ease-in-out infinite' : 'none',
            }}>
            <Sparkles size={12}/>
            {isDone ? 'Re-grade' : 'Grade'}
          </button>
        )}
      </header>

      {/* Progress bar */}
      <div style={{ height:2, flexShrink:0, background:C.border, overflow:'hidden', position:'relative' }}>
        {listLoading && (
          <div style={{ position:'absolute', top:0, height:'100%', background:`linear-gradient(90deg,transparent,${C.accent},transparent)`, animation:'sweep 1.2s ease-in-out infinite' }}/>
        )}
        {!listLoading && isGrading && (
          <div style={{ height:'100%', width:`${Math.max(4, gradeProgress * 100)}%`, background:`linear-gradient(90deg,${C.accent},oklch(70% .19 205))`, transition:'width .25s ease' }}/>
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
                  style={{ display:'flex', alignItems:'center', gap:5, padding:'3px 9px', borderRadius:5, cursor:'pointer', transition:'all .15s',
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
            style={{ display:'flex', alignItems:'center', gap:5, padding:'3px 9px', borderRadius:5, cursor:'pointer', transition:'all .15s',
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
                onSelect={id => { setSelId(id); setLoupeMode('loupe'); }}
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
              {sel ? (
                <>
                  <img
                    key={sel.path}
                    src={photoUrl(sel.path)}
                    alt=""
                    style={{ maxWidth:'100%', maxHeight:'100%', objectFit:'contain', display:'block', userSelect:'none', animation:'fadeIn .2s ease-out', outline: selectedIds.has(selId ?? '') ? `3px solid ${C.accent}` : 'none', outlineOffset:'-3px', transition:'outline .15s' }}
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
                        style={{ position:'absolute', bottom:16, left:16, display:'flex', alignItems:'center', gap:6, padding:'6px 12px', borderRadius:20, cursor:'pointer', transition:'all .15s', background:isSel ? C.accent : 'rgba(0,0,0,.6)', backdropFilter:'blur(12px)', border:`1px solid ${isSel ? C.accent : 'rgba(255,255,255,.12)'}`, color:'#fff', fontSize:12, fontWeight:700 }}>
                        <div style={{ width:14, height:14, borderRadius:3, flexShrink:0, background:isSel?'#fff':'transparent', border:`1.5px solid ${isSel?C.accent:'rgba(255,255,255,.6)'}`, display:'flex', alignItems:'center', justifyContent:'center' }}>
                          {isSel && <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke={C.accent} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20,6 9,17 4,12"/></svg>}
                        </div>
                        {isSel ? 'Selected' : 'Select'}
                      </button>
                    );
                  })()}
                  {/* Floating action bar */}
                  {selectedIds.size > 0 && (
                    <div style={{ position:'absolute', bottom:16, left:'50%', transform:'translateX(-50%) translateX(40px)', display:'flex', alignItems:'center', gap:10, background:C.surf, border:`1px solid ${C.bdr2}`, borderRadius:12, padding:'10px 18px', boxShadow:'0 8px 40px rgba(0,0,0,.7)', backdropFilter:'blur(12px)', zIndex:50, whiteSpace:'nowrap', animation:'slideUp .22s cubic-bezier(.2,0,0,1)' }}>
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
            <div
              onMouseDown={onResizeDown}
              style={{ width:3, cursor:'col-resize', flexShrink:0, background:'transparent', transition:'background .15s' }}
              onMouseEnter={e => (e.currentTarget.style.background = 'oklch(64% .19 248 / .3)')}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            />

            {/* Right panel */}
            <div style={{ width:rightW, flexShrink:0, background:C.surf, borderLeft:`1px solid ${C.border}`, display:'flex', flexDirection:'column', overflow:'hidden' }}>

              {/* Thumbnail */}
              {sel && (
                <div style={{ flexShrink:0, position:'relative', aspectRatio:'3/2', background:C.bg, overflow:'hidden' }}>
                  <img key={sel.path} src={thumbUrl(sel.path)} alt=""
                    style={{ width:'100%', height:'100%', objectFit:'cover', display:'block', animation:'fadeIn .2s' }}/>
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
                      style={{ display:'flex', alignItems:'center', gap:4, padding:'4px 7px', borderRadius:5, background:copied ? C.sLow : C.surf2, border:`1px solid ${C.bdr2}`, color:copied ? C.strong : C.text3, fontSize:11, cursor:'pointer', transition:'all .15s' }}>
                      <Copy size={10}/>{copied ? 'Copied!' : ''}
                    </button>
                  </div>
                  <StarRating stars={sel.stars ?? 0} onSet={n => handleSetStars(sel.id, n)}/>
                </div>
              )}

              {/* Tabs */}
              {sel && (
                <div style={{ flexShrink:0, display:'flex', borderBottom:`1px solid ${C.border}` }}>
                  {(isDone
                    ? [['breakdown','Breakdown'],['analysis','Analysis'],['exif','EXIF']]
                    : [['exif','EXIF']]
                  ).map(([id, label]) => (
                    <button key={id} onClick={() => setInfoTab(id as any)}
                      style={{ flex:1, height:34, fontSize:12.5, fontWeight:600, cursor:'pointer', background:'none', border:'none', borderBottom:`2px solid ${infoTab===id ? C.accent : 'transparent'}`, color:infoTab===id ? C.accent : C.text3, transition:'all .15s', letterSpacing:'.03em', marginBottom:-1 }}>
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
                    : <p style={{ fontSize:13, color:C.text3, lineHeight:1.6 }}>Select a photo to view EXIF data.</p>
                )}
                {infoTab === 'analysis' && (
                  isGrading ? (
                    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height:'100%', gap:8 }}>
                      <span style={{ width:20, height:20, borderRadius:'50%', border:`2px solid ${C.accent}`, borderTopColor:'transparent', animation:'spin .8s linear infinite', display:'inline-block' }}/>
                      <p style={{ fontSize:13, color:C.text3 }}>Analysing…</p>
                    </div>
                  ) : isGraded ? (
                    <p style={{ fontSize:13, color:C.text2, lineHeight:1.85, animation:'fadeIn .25s' }}>{sel?.critique || 'No critique available.'}</p>
                  ) : (
                    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:8, padding:'20px 0' }}>
                      <Layers size={24} strokeWidth={1} style={{ color:C.text3 }}/>
                      <p style={{ fontSize:13, color:C.text3, textAlign:'center', lineHeight:1.6 }}>Grade your folder to see analysis.</p>
                    </div>
                  )
                )}
                {infoTab === 'breakdown' && (
                  isGraded ? (
                    <div style={{ display:'flex', flexDirection:'column', gap:12, animation:'fadeIn .25s' }}>
                      {Object.entries(sel?.breakdown || {}).filter(([k, v]) => typeof v === 'number' && isFinite(v as number) && !['Median_Score','Best_Score','Applied_Preset','Best_Preset'].includes(k)).map(([k, v]) => {
                        const pct = Math.round((v as number)*100);
                        const col = pct>59 ? C.strong : pct>40 ? C.mid : C.weak;
                        return (
                          <div key={k}>
                            <div style={{ display:'flex', justifyContent:'space-between', marginBottom:5 }}>
                              <span style={{ fontSize:12, color:C.text2, fontWeight:500 }}>{k}</span>
                              <span style={{ fontSize:12, fontWeight:700, color:col, fontVariantNumeric:'tabular-nums' }}>{pct}%</span>
                            </div>
                            <div style={{ height:2.5, background:C.bdr2, borderRadius:2, overflow:'hidden' }}>
                              <div style={{ height:'100%', width:`${pct}%`, background:col, borderRadius:2, transition:'width .6s cubic-bezier(.2,0,0,1)' }}/>
                            </div>
                          </div>
                        );
                      })}
                      <div style={{ borderTop:`1px solid ${C.bdr2}`, paddingTop:10, marginTop:2 }}>
                        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                          <span style={{ fontSize:12, color:C.text2, fontWeight:600, letterSpacing:'.03em' }}>Total Average</span>
                          <span style={{ fontSize:14, fontWeight:800, color: Math.round((sel?.score||0)*100)>59 ? C.strong : Math.round((sel?.score||0)*100)>40 ? C.mid : C.weak, fontVariantNumeric:'tabular-nums' }}>{Math.round((sel?.score||0)*100)}%</span>
                        </div>
                        <div style={{ height:3, background:C.bdr2, borderRadius:2, overflow:'hidden', marginTop:6 }}>
                          <div style={{ height:'100%', width:`${Math.round((sel?.score||0)*100)}%`, background: Math.round((sel?.score||0)*100)>59 ? C.strong : Math.round((sel?.score||0)*100)>40 ? C.mid : C.weak, borderRadius:2, transition:'width .6s cubic-bezier(.2,0,0,1)' }}/>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:8, padding:'20px 0' }}>
                      <Layers size={24} strokeWidth={1} style={{ color:C.text3 }}/>
                      <p style={{ fontSize:13, color:C.text3, textAlign:'center', lineHeight:1.6 }}>Grade your folder to see breakdown.</p>
                    </div>
                  )
                )}
              </div>

            </div>

            </>)}
          </div>

          {/* ── Filmstrip (loupe mode only) ─────────────────────── */}
          {loupeMode === 'loupe' && (
          <div style={{ flexShrink:0, background:C.surf, borderTop:`1px solid ${C.border}`, display:'flex', flexDirection:'column' }}>
            <div style={{ height:20, flexShrink:0, display:'flex', alignItems:'center', justifyContent:'space-between', padding:'0 12px', borderBottom:`1px solid ${C.border}` }}>
              <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                <span style={{ fontSize:10.5, color:C.text3, fontWeight:600, letterSpacing:'.08em', textTransform:'uppercase' }}>Library</span>
                {/* Tweaks toggle */}
                <button title="Filmstrip settings" onClick={() => setShowTweaks(v => !v)}
                  style={{ display:'flex', alignItems:'center', justifyContent:'center', width:18, height:16, cursor:'pointer', background:showTweaks ? C.surf3 : 'transparent', color:showTweaks ? C.accent : C.text3, border:'none', borderRadius:3, transition:'all .15s' }}>
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
                    style={{ position:'relative', width:28, height:16, borderRadius:8, border:'none', cursor:'pointer', padding:0, background:showFilename ? C.accent : C.bdr2, transition:'background .15s' }}>
                    <span style={{ position:'absolute', top:2, left:showFilename ? 13 : 2, width:12, height:12, borderRadius:'50%', background:'#fff', transition:'left .15s', boxShadow:'0 1px 2px rgba(0,0,0,.3)' }}/>
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
        /* ── Duplicates view ───────────────────────────────────── */
        (() => {
          // Build groups: cluster_id → [photo, ...], best first
          const byCluster: Record<number, any[]> = {};
          for (const p of photos) {
            if (p.cluster_id < 0) continue;
            (byCluster[p.cluster_id] ??= []).push(p);
          }
          const groups = Object.values(byCluster)
            .map(g => {
              const best = g.find(p => (p.sim_flag||'').includes('Best')) ?? g[0];
              const rest = g.filter(p => p !== best);
              return { best, rest, all: [best, ...rest] };
            })
            .sort((a, b) => b.all.length - a.all.length);
          const totalDups = groups.reduce((s, g) => s + g.rest.length, 0);
          const redactGroup = (g: {rest: any[]}) => {
            setRedacted(prev => { const next = new Set(prev); g.rest.forEach(p => next.add(p.path)); return next; });
          };
          const redactAll = () => {
            setRedacted(prev => {
              const next = new Set(prev);
              groups.forEach(g => g.rest.forEach(p => next.add(p.path)));
              return next;
            });
          };
          const redactedInGroups = groups.reduce((s, g) => s + g.rest.filter(p => redacted.has(p.path)).length, 0);

          return (
            <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', background:C.bg }}>
              {/* Header */}
              <div style={{ flexShrink:0, display:'flex', alignItems:'center', gap:10, padding:'10px 18px', borderBottom:`1px solid ${C.border}`, background:C.surf }}>
                <span style={{ fontSize:15, fontWeight:700 }}>Similar / Duplicates</span>
                <span style={{ fontSize:12, color:C.text3 }}>{groups.length} group{groups.length!==1?'s':''}</span>
                {nicheRec?.preset && (
                  <span style={{ fontSize:11, color:C.accent, background:C.aLow, border:`1px solid ${C.aBdr}`, borderRadius:4, padding:'2px 8px', fontWeight:600 }}>
                    {nicheRec.preset}
                  </span>
                )}
                <div style={{ marginLeft:'auto', display:'flex', alignItems:'center', gap:8 }}>
                  {redactedInGroups > 0 && (
                    <button onClick={() => setRedacted(new Set())}
                      style={{ fontSize:13, color:C.text3, padding:'0 10px', height:28, borderRadius:6, border:`1px solid ${C.bdr2}`, background:'transparent', cursor:'pointer' }}>
                      Restore {redactedInGroups} hidden
                    </button>
                  )}
                  {totalDups > 0 && (
                    <button onClick={redactAll}
                      style={{ display:'flex', alignItems:'center', gap:6, fontSize:13, fontWeight:700, color:'#fff', padding:'0 14px', height:30, borderRadius:6, border:'none', background:C.strong, cursor:'pointer' }}>
                      <CheckSquare size={13}/>
                      Keep all best
                    </button>
                  )}
                </div>
              </div>

              {/* Groups */}
              <div style={{ flex:1, overflowY:'auto', padding:'14px 16px', display:'flex', flexDirection:'column', gap:12 }}>
                {groups.map((g, gi) => (
                  <div key={gi} style={{ background:C.surf, border:`1px solid ${C.border}`, borderRadius:8 }}>
                    {/* Group label */}
                    <div style={{ padding:'6px 10px 6px 12px', borderBottom:`1px solid ${C.border}`, display:'flex', alignItems:'center', gap:8 }}>
                      <span style={{ fontSize:11, fontWeight:700, color:C.text3, letterSpacing:'.07em', textTransform:'uppercase' }}>Group {gi + 1}</span>
                      <span style={{ fontSize:11, color:C.text3 }}>{g.all.length} similar</span>
                      {g.rest.some(p => redacted.has(p.path)) ? (
                        <button onClick={() => setRedacted(prev => { const next = new Set(prev); g.rest.forEach(p => next.delete(p.path)); return next; })}
                          style={{ marginLeft:'auto', fontSize:12, color:C.text3, padding:'2px 8px', borderRadius:4, border:`1px solid ${C.bdr2}`, background:'transparent', cursor:'pointer' }}>
                          Restore
                        </button>
                      ) : (
                        <button onClick={() => redactGroup(g)}
                          style={{ marginLeft:'auto', display:'flex', alignItems:'center', gap:5, fontSize:12, fontWeight:600, color:C.strong, padding:'2px 8px', borderRadius:4, border:`1px solid oklch(65% .17 148 / .35)`, background:'oklch(65% .17 148 / .10)', cursor:'pointer' }}>
                          <CheckSquare size={11}/>
                          Keep best
                        </button>
                      )}
                    </div>

                    {/* Thumbnail grid — wraps, no horizontal scroll */}
                    <div style={{ padding:10, display:'flex', flexWrap:'wrap', gap:8 }}>
                      {g.all.map((p) => {
                        const isBest = p === g.best;
                        const gradeLabel = gl(p.grade);
                        const gradeColor = gc(p.grade);
                        const gradeBg    = gLow(p.grade);
                        return (
                          <div key={p.id} onClick={() => { setMainTab('gallery'); setSelId(p.id); setLoupeMode('loupe'); }}
                            style={{
                              width:160, flexShrink:0, cursor:'pointer', borderRadius:6, overflow:'hidden',
                              border: isBest ? `2px solid ${gradeColor}` : `1px solid ${C.border}`,
                              background: isBest ? gradeBg : C.bg,
                              opacity: redacted.has(p.path) ? 0.35 : 1,
                              transition:'opacity .2s',
                            }}>
                            <div style={{ position:'relative', width:'100%', height:110, overflow:'hidden' }}>
                              <img src={thumbUrl(p.path)} alt="" loading="lazy" decoding="async"
                                style={{ width:'100%', height:'100%', objectFit:'cover', display:'block' }}/>
                              {/* Grade label */}
                              {gradeLabel !== 'Pending' && (
                                <div style={{
                                  position:'absolute', top:5, left:5,
                                  background: `${gradeColor}dd`,
                                  borderRadius:3, padding:'2px 6px', fontSize:9, fontWeight:700, color:'#fff', letterSpacing:'.07em',
                                }}>
                                  {gradeLabel}
                                </div>
                              )}
                              <div style={{ position:'absolute', top:5, right:5, background:'rgba(0,0,0,.65)', borderRadius:3, padding:'2px 5px', fontSize:9, color:C.text2 }}>
                                {(p.score*100).toFixed(0)}
                              </div>
                            </div>
                            <div style={{ padding:'5px 8px' }}>
                              <p style={{ fontSize:10.5, color: isBest ? gradeColor : C.text3, fontWeight: isBest ? 700 : 400, margin:0, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>
                                {p.path.split(/[\\/]/).pop()}
                              </p>
                              <p style={{ fontSize:10, color: gradeColor, margin:'2px 0 0', opacity: isBest ? 1 : 0.7 }}>
                                {isBest ? 'Best of group' : gradeLabel}
                              </p>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}

                {groups.length === 0 && (
                  <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', flex:1, gap:10, color:C.text3, paddingTop:60 }}>
                    <ImageOff size={32} strokeWidth={1}/>
                    <p style={{ fontSize:14 }}>No duplicates found in this folder.</p>
                  </div>
                )}
              </div>
            </div>
          );
        })()

      ) : (
        /* ── Sequence view ─────────────────────────────────────── */
        <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', background:C.bg }}>
          {/* Toolbar */}
          <div style={{ flexShrink:0, display:'flex', alignItems:'center', justifyContent:'space-between', padding:'10px 18px', borderBottom:`1px solid ${C.border}`, background:C.surf }}>
            <div style={{ display:'flex', alignItems:'center', gap:10 }}>
              <span style={{ fontSize:15, fontWeight:700 }}>Story Sequence</span>
              {subjType && <span style={{ fontSize:12, color:C.text3 }}>{subjType}</span>}
              {carousel.length > 0 && <span style={{ fontSize:13, color:C.text3 }}>{carousel.length} frames</span>}
            </div>
            <div style={{ display:'flex', alignItems:'center', gap:8 }}>
              {carousel.some(c => c.stars > 0) && (
                <div style={{ position:'relative', flexShrink:0 }}>
                  <button onClick={() => setShowStarSort(v => !v)}
                    style={{ display:'flex', alignItems:'center', gap:5, padding:'5px 11px', background:showStarSort ? C.surf3 : C.surf2, border:`1px solid ${showStarSort ? C.aBdr : C.bdr2}`, borderRadius:7, color:showStarSort ? C.accent : C.text2, fontSize:13, fontWeight:600, cursor:'pointer' }}>
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="oklch(70% .18 72)" strokeWidth="0"><polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/></svg>
                    Sort by Stars ▾
                  </button>
                  {showStarSort && (
                    <>
                      {/* backdrop */}
                      <div style={{ position:'fixed', inset:0, zIndex:49 }} onClick={() => setShowStarSort(false)}/>
                      <div style={{ position:'absolute', top:'calc(100% + 6px)', left:0, zIndex:50, background:C.surf, border:`1px solid ${C.bdr2}`, borderRadius:8, overflow:'hidden', boxShadow:'0 8px 32px rgba(0,0,0,.6)', minWidth:130 }}>
                        {[5,4,3,2,1].map(n => (
                          <button key={n} onClick={() => { handleSortByStars(n); setShowStarSort(false); }}
                            style={{ width:'100%', display:'flex', alignItems:'center', gap:8, padding:'8px 12px', background:'transparent', border:'none', cursor:'pointer', transition:'background .1s' }}
                            onMouseEnter={e => (e.currentTarget.style.background = C.surf2)}
                            onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                            <div style={{ display:'flex', gap:2 }}>
                              {[1,2,3,4,5].map(s => (
                                <svg key={s} width="12" height="12" viewBox="0 0 24 24"
                                  fill={s <= n ? 'oklch(70% .18 72)' : 'oklch(30% .04 72)'} stroke="none">
                                  <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
                                </svg>
                              ))}
                            </div>
                            <span style={{ fontSize:12, color:C.text2, fontWeight:600 }}>{n} star{n !== 1 ? 's' : ''} first</span>
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}
              {/* Min-star pool filter (sequence toolbar) */}
              <div style={{ display:'flex', alignItems:'center', gap:3, background:C.surf2, border:`1px solid ${C.bdr2}`, borderRadius:7, padding:'3px 6px' }}>
                <span style={{ fontSize:11, color:C.text3, marginRight:2, whiteSpace:'nowrap' }}>Pool:</span>
                {([0,1,2,3,4,5] as const).map(n => (
                  <button key={n} onClick={() => setSeqMinStars(n)}
                    style={{ padding:'2px 6px', borderRadius:5, fontSize:11, fontWeight:seqMinStars===n?700:500, cursor:'pointer',
                      background: seqMinStars===n ? C.aLow : 'transparent',
                      border: `1px solid ${seqMinStars===n ? C.aBdr : 'transparent'}`,
                      color: seqMinStars===n ? C.accent : C.text3, transition:'all .12s' }}>
                    {n === 0 ? 'Any' : `${n}★+`}
                  </button>
                ))}
              </div>
              <button onClick={handleGenerate} disabled={loading}
                style={{ display:'flex', alignItems:'center', gap:6, padding:'5px 12px', background:C.surf2, border:`1px solid ${C.bdr2}`, borderRadius:7, color:C.text2, fontSize:13, fontWeight:600, cursor:'pointer' }}>
                <RefreshCw size={12}/> Regenerate
              </button>
              <button onClick={handleExport} disabled={carousel.length<5}
                style={{ display:'flex', alignItems:'center', gap:6, padding:'5px 12px', background:'oklch(35% .12 295 / .2)', border:'1px solid oklch(45% .12 295 / .3)', borderRadius:7, color:'oklch(75% .12 295)', fontSize:13, fontWeight:600, cursor:'pointer', opacity:carousel.length<5?0.4:1 }}>
                <FileDown size={12}/> Export
              </button>
              <button onClick={handleSave} disabled={carousel.length===0}
                style={{ display:'flex', alignItems:'center', gap:6, padding:'5px 14px', background:C.aLow, border:`1px solid ${C.aBdr}`, borderRadius:7, color:C.accent, fontSize:13, fontWeight:700, cursor:'pointer', opacity:!carousel.length?0.4:1 }}>
                <Flag size={12}/> Save Story
              </button>
            </div>
          </div>

          {/* Sequence cards */}
          {carousel.length === 0 ? (
            <div style={{ flex:1, display:'flex', alignItems:'center', justifyContent:'center', flexDirection:'column', gap:12, color:C.text3 }}>
              <Layers size={32} strokeWidth={1}/>
              <p style={{ fontSize:14 }}>No sequence yet — click Generate Sequence</p>
            </div>
          ) : (
            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
              <SortableContext items={carousel.map(c => c.path)} strategy={verticalListSortingStrategy}>
                <div style={{ flex:1, overflowY:'auto', padding:24 }}>
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(5,1fr)', gap:16, maxWidth:1100, margin:'0 auto' }}>
                    {carousel.map((c, i) => {
                      const isSel  = sel?.path === c.path;
                      const isUsed = allUsedPaths.has(c.path);
                      return (
                        <SortableItem key={c.path} id={c.path}>
                          <div onClick={() => jumpToPhoto(c.path)}
                            style={{ background:C.surf, borderRadius:10, overflow:'hidden', cursor:'pointer', border:`1px solid ${isSel?C.accent:C.border}`, boxShadow:`0 2px 16px rgba(0,0,0,.4)`, display:'flex', flexDirection:'column' }}>
                            <div style={{ position:'relative', aspectRatio:'2/3' }}>
                              <img src={thumbUrl(c.path)} alt="" style={{ width:'100%', height:'100%', objectFit:'cover', display:'block' }}/>
                              <div style={{ position:'absolute', top:8, left:8, background:'rgba(0,0,0,.72)', backdropFilter:'blur(8px)', borderRadius:5, padding:'3px 8px', fontSize:12, fontWeight:700, color:'#fff' }}>{i+1}</div>
                              {c.stars > 0 && (
                                <div style={{ position:'absolute', bottom:8, left:8, display:'flex', gap:2 }}>
                                  {[1,2,3,4,5].map(s => (
                                    <svg key={s} width="9" height="9" viewBox="0 0 24 24"
                                      fill={s <= c.stars ? 'oklch(70% .18 72)' : 'none'}
                                      stroke={s <= c.stars ? 'oklch(70% .18 72)' : 'rgba(255,255,255,.3)'} strokeWidth="2">
                                      <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
                                    </svg>
                                  ))}
                                </div>
                              )}
                            </div>
                            <div style={{ padding:'9px 11px' }}>
                              <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
                                <p style={{ fontSize:12, fontWeight:600, color:C.text, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', flex:1 }}>{c.path.split(/[\\/]/).pop()}</p>
                                {isUsed && (
                                  <div style={{ flexShrink:0, marginLeft:6, display:'flex', alignItems:'center', gap:3, background:C.aLow, borderRadius:4, padding:'2px 6px' }}>
                                    <Flag size={8} style={{ color:C.accent, flexShrink:0 }}/>
                                    <span style={{ fontSize:10, fontWeight:700, color:C.accent }}>USED</span>
                                  </div>
                                )}
                              </div>
                              {c.rationale && c.rationale !== 'Strong candidate.' && (
                                <p style={{ fontSize:10.5, color:C.text3, lineHeight:1.55, marginTop:4, overflow:'hidden', display:'-webkit-box', WebkitLineClamp:2, WebkitBoxOrient:'vertical' as any }}>{c.rationale}</p>
                              )}
                            </div>
                          </div>
                        </SortableItem>
                      );
                    })}
                  </div>
                  <p style={{ textAlign:'center', fontSize:12, color:C.text3, marginTop:16 }}>Drag cards to reorder · Click to view in Gallery</p>

                  {/* ── Sequence narrative ── */}
                  {sequenceNarrative && (
                    <div style={{ maxWidth:1100, margin:'20px auto 0', padding:'16px 20px', background:C.surf, border:`1px solid ${C.border}`, borderRadius:10 }}>
                      <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:10 }}>
                        <span style={{ fontSize:11, fontWeight:700, letterSpacing:'.08em', textTransform:'uppercase', color:C.accent }}>Sequence Narrative</span>
                        {nicheRec && (
                          <span style={{ fontSize:11, color:C.text3, background:C.surf2, border:`1px solid ${C.bdr2}`, borderRadius:4, padding:'1px 6px' }}>{nicheRec.preset}</span>
                        )}
                      </div>
                      <p style={{ fontSize:14, color:C.text2, lineHeight:1.85 }}>{sequenceNarrative}</p>
                    </div>
                  )}
                </div>
              </SortableContext>
            </DndContext>
          )}

          {/* ── Saved stories panel ──────────────────────────────── */}
          {saved.length > 0 && (
            <div style={{ flexShrink:0, borderTop:`1px solid ${C.border}`, background:C.surf }}>
              <div style={{ display:'flex', alignItems:'center', padding:'6px 18px', borderBottom:`1px solid ${C.border}` }}>
                <span style={{ fontSize:11, fontWeight:700, letterSpacing:'.08em', textTransform:'uppercase', color:C.text3 }}>Saved Stories</span>
                <span style={{ fontSize:11, color:C.text3, marginLeft:8 }}>{saved.length}</span>
              </div>
              <div style={{ display:'flex', gap:10, padding:'10px 18px', overflowX:'auto', overflowY:'hidden' }}>
                {saved.map((s, idx) => (
                  <div key={idx}
                    style={{ flexShrink:0, width:160, background:C.surf2, borderRadius:8, border:`1px solid ${C.bdr2}`, overflow:'hidden', display:'flex', flexDirection:'column', cursor:'pointer', transition:'border .15s' }}
                    onClick={() => { setCarousel(s.sequence); notify(`Loaded "${s.name}"`, 'info'); }}
                    onMouseEnter={e => (e.currentTarget.style.border = `1px solid ${C.aBdr}`)}
                    onMouseLeave={e => (e.currentTarget.style.border = `1px solid ${C.bdr2}`)}>
                    {/* Thumbnail strip */}
                    <div style={{ display:'flex', height:52, overflow:'hidden' }}>
                      {s.sequence.slice(0, 5).map((c: any, j: number) => (
                        <div key={j} style={{ flex:1, position:'relative', overflow:'hidden', borderRight: j < 4 ? `1px solid ${C.border}` : 'none' }}>
                          <img src={thumbUrl(c.path)} alt="" decoding="async"
                            style={{ width:'100%', height:'100%', objectFit:'cover', display:'block' }}/>
                        </div>
                      ))}
                    </div>
                    {/* Label row */}
                    <div style={{ display:'flex', alignItems:'center', padding:'6px 9px', gap:4 }}>
                      <div style={{ flex:1 }}>
                        <p style={{ fontSize:13, fontWeight:700, color:C.text }}>{s.name}</p>
                        <p style={{ fontSize:11, color:C.text3, marginTop:1 }}>{s.sequence.length} frames</p>
                      </div>
                      <button
                        onClick={e => { e.stopPropagation(); handleDeleteSaved(idx); }}
                        style={{ display:'flex', alignItems:'center', justifyContent:'center', width:18, height:18, borderRadius:4, border:`1px solid ${C.bdr2}`, background:'transparent', color:C.text3, cursor:'pointer', flexShrink:0 }}>
                        <X size={10}/>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

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
                onClick={() => { setFolder(bPath); setPhotos([]); setSelId(null); setShowBrowser(false); }}
                disabled={bImages.length===0}
                style={{ flexShrink:0, padding:'4px 12px', fontSize:13, fontWeight:600, background:'#2563eb', color:'#fff', borderRadius:7, border:'none', cursor:bImages.length>0?'pointer':'not-allowed', opacity:bImages.length>0?1:0.4 }}>
                Use Folder{bImages.length>0 ? ` (${bImages.length})` : ''}
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
                          {bFolders.map(f => (
                            <button key={f} onClick={() => { setBPath(f); loadBrowser(f); }}
                              style={{ display:'flex', alignItems:'center', gap:8, padding:'8px 12px', background:'#161b22', border:'1px solid #252d38', borderRadius:8, cursor:'pointer', textAlign:'left' }}>
                              <FolderOpen size={13} style={{ color:'#60a5fa', flexShrink:0 }}/>
                              <span style={{ fontSize:13, color:'#c0c0d0', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{f.split(/[\\/]/).pop()}</span>
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
