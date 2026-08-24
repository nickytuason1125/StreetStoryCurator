import { Layers, Eye, EyeOff, Wand2, Copy, Download, RefreshCw } from 'lucide-react';
import { Button } from '../ui/Button';
import { Thumb } from '../photo/Thumb';
import { StarRating } from '../ui/StarRating';
import { ExifPanel } from '../ExifPanel';
import { T, gradeLabel, formatScore, gradeRule } from '../../theme/tokens';
import { API, photoUrl } from '../../lib/api';
import { gc } from '../../lib/grading';
import { cn } from '../../lib/cn';
import { regionGuide, tierColor, tierIcon, tierHeat } from '../../lib/regions';
import type { RegionTier } from '../../lib/regions';
import { aspectDim } from '../../lib/aspects';

/* â”€â”€ AnalysisPanel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
 * The loupe's right rail: thumbnail header with grade pill, rating +
 * telemetry, the vision-critique stack (jury / VLM / deep critique,
 * evidence checklist, spatial anchors) and the EXIF block. Extracted
 * verbatim from App.tsx during the views split â€” props carry the same
 * names as the App state they wrap, so the JSX is untouched. */
export function AnalysisPanel({
  sel, selId, setSelId, photos, rightW, isGraded, isDone,
  isAuditModeActive, setIsAuditModeActive,
  critTrigger, setCritTrigger, juryCritique, juryLoading, handleJuryCritique,
  parseCritique, deepCritique, setDeepCritique, deepCritiqueLoading, setDeepCritiqueLoading,
  reasoningOverlayUrl, buildReasoningFromBreakdown,
  infoTab, setInfoTab, selectedIds, setSelectedIds,
  handleCopyPath, handleSetStars, setMainTab, copied,
  handleGenerate, handleCreateFromSelection,
  hasPrev, hasNext, selIdx, filteredPhotos,
}: {
  sel: any; selId: string | null; setSelId: (id: string) => void;
  photos: any[]; rightW: number; isGraded: boolean; isDone: boolean;
  isAuditModeActive: boolean; setIsAuditModeActive: React.Dispatch<React.SetStateAction<boolean>>;
  critTrigger: string | null; setCritTrigger: (v: string | null) => void;
  juryCritique: string | null; juryLoading: boolean; handleJuryCritique: (path: string) => void;
  parseCritique: any; deepCritique: any; setDeepCritique: (v: any) => void;
  deepCritiqueLoading: boolean; setDeepCritiqueLoading: (v: boolean) => void;
  reasoningOverlayUrl: string | null; buildReasoningFromBreakdown: any;
  infoTab: string; setInfoTab: (v: string) => void;
  selectedIds: Set<string>; setSelectedIds: React.Dispatch<React.SetStateAction<Set<string>>>;
  handleCopyPath: (path: string) => void; handleSetStars: (id: string, stars: number) => void;
  setMainTab: (v: string) => void; copied: boolean;
  handleGenerate: () => void; handleCreateFromSelection: () => void;
  hasPrev: boolean; hasNext: boolean; selIdx: number; filteredPhotos: any[];
}) {
  return (
    <>
            {photos.length > 0 && <div style={{ width:rightW, flexShrink:0, background:T.surface, borderLeft:`1px solid ${T.line}`, display:'flex', flexDirection:'column', overflow:'hidden' }}>

              {/* Thumbnail */}
              {sel && (
                <div style={{ flexShrink:0, position:'relative', aspectRatio:'3/2', background:T.ground, overflow:'hidden' }}>
                  <Thumb key={sel.path} path={sel.path} eager
                    style={{ width:'100%', height:'100%', objectFit:'cover', display:'block' }}/>
                  {isGraded && (
                    <div style={{ position:'absolute', inset:0, background:`linear-gradient(to top,${T.scrim} 0%,transparent 55%)`, display:'flex', alignItems:'flex-end', padding:'10px 12px' }}>
                      <div style={{ display:'flex', alignItems:'center', gap:6, background:T.scrim, backdropFilter:'blur(8px)', borderRadius:'var(--r-md)', padding:'6px 12px', border:`1px solid ${gc(sel.grade)}` }}>
                        <div style={{ width:8, height:8, borderRadius:'var(--r-round)', background:gc(sel.grade), flexShrink:0 }}/>
                        <span style={{ fontSize:'var(--text-md)', fontWeight:500, color:T.ink }}>{gradeLabel(sel.grade)}</span>
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
                      {(['Strong','Mid','Weak'] as const).map(g => {
                        const _sc = sel.score ?? 0;
                        const derivedGrade = gradeLabel(_sc >= 0.60 ? 'Strong' : _sc >= 0.41 ? 'Mid' : 'Weak');
                        const isActive = derivedGrade === g;
                        const col = g.includes('Strong') ? T.gradeStrong : g.includes('Mid') ? T.ink2 : T.gradeWeak;
                        return (
                          <div key={g}
                            style={{ flex:1, padding:'3px 0', borderRadius:'var(--r-sm)', fontSize:'var(--text-xs)', fontWeight:600,
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
                            fontWeight:600, fontSize:'var(--text-xs)', letterSpacing:'.03em',
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
                            <span style={{ fontSize:'var(--text-sm)', fontWeight:600, letterSpacing:'.08em', color:gradeCol }}>{tierWord.toUpperCase()}</span>
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
                                      <span style={{ fontSize:'var(--text-xs)', fontWeight:500,
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
                                  <span style={{ fontSize:'var(--text-xs)', fontWeight:500, letterSpacing:'.08em', color:T.ink3 }}>NARRATIVE</span>
                                  <p style={{ fontSize:'var(--text-xs)', color:T.ink2, lineHeight:1.65, margin:'4px 0 0' }}>{_dnarr}</p>
                                </div>
                              )}
                              {_dgeo && (
                                <div style={{ animation:'fadeIn .4s cubic-bezier(.2,0,0,1)' }}>
                                  <span style={{ fontSize:'var(--text-xs)', fontWeight:500, letterSpacing:'.08em', color:T.ink3 }}>GEOMETRY</span>
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
                                  <span style={{ fontSize:'var(--text-xs)', fontWeight:500, letterSpacing:'.08em', color:T.ink3 }}>SPATIAL ANCHORS</span>
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
                                          <span style={{ fontSize:'var(--text-xs)', fontWeight:500, letterSpacing:'.06em',
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
                              style={{ display:'flex', alignItems:'center', gap:6, padding:'7px 14px', borderRadius:'var(--r-md)', background:T.raised, border:`1px solid ${T.lineStrong}`, color:T.ink2, fontSize:'var(--text-sm)', fontWeight:600, cursor: juryLoading ? 'wait' : 'pointer', alignSelf:'flex-start' }}>
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
                                <span className={cn('text-sm',
                                                    _isPrimary ? 'text-ink' : 'text-ink-3')}>
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
                                <span style={{ fontSize:'var(--text-xs)', fontWeight:500, letterSpacing:'.09em',
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
                                        <span style={{ fontSize:'var(--text-xs)', fontWeight:500, letterSpacing:'.07em',
                                          color:T.ink3, minWidth:62, textTransform:'uppercase' }}>{label}</span>
                                        <span style={{ fontSize:'var(--text-xs)', fontWeight:600, color:col }}>{value}</span>
                                        {isLimit && (
                                          <span style={{ marginLeft:'auto', fontSize:'var(--text-xs)', fontWeight:600,
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
    </>
  );
}


