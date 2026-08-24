import { RefreshCw, Layers, Eye } from 'lucide-react';
import { Button } from '../ui/Button';
import { Thumb } from '../photo/Thumb';
import { T } from '../../theme/tokens';
import { API, photoUrl } from '../../lib/api';
import { gc } from '../../lib/grading';
import { regionGuide, tierColor, tierIcon, tierHeat } from '../../lib/regions';
import type { RegionTier } from '../../lib/regions';

/* â”€â”€ LoupeStage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
 * The centre preview: the photograph, the judge's-eye overlay and its
 * toggle, the critique heatmap + legend, the criteria annotations, the
 * prev/next arrows, the select toggle and the floating action bar.
 * Extracted verbatim from App.tsx during the views split â€” props carry
 * the same names as the App state they wrap, so the JSX is untouched. */
export function LoupeStage({
  sel, selId, setSelId, loupePreviewFailed, setLoupePreviewFailed,
  loupeRetry, setLoupeRetry, photoNatDims, setPhotoNatDims,
  selectedIds, setSelectedIds, showEyeOverlay, setShowEyeOverlay,
  showHeatmap, critTrigger, heatmapB64, isAuditModeActive, isGraded,
  hasPrev, hasNext, selIdx, filteredPhotos,
  handleCreateFromSelection, handleGenerate,
}: {
  sel: any; selId: string | null; setSelId: (id: string) => void;
  loupePreviewFailed: boolean; setLoupePreviewFailed: (v: boolean) => void;
  loupeRetry: number; setLoupeRetry: React.Dispatch<React.SetStateAction<number>>;
  photoNatDims: { w: number; h: number } | null; setPhotoNatDims: (d: { w: number; h: number }) => void;
  selectedIds: Set<string>; setSelectedIds: React.Dispatch<React.SetStateAction<Set<string>>>;
  showEyeOverlay: boolean; setShowEyeOverlay: React.Dispatch<React.SetStateAction<boolean>>;
  showHeatmap: boolean; critTrigger: string | null; heatmapB64: string | null;
  isAuditModeActive: boolean; isGraded: boolean;
  hasPrev: boolean; hasNext: boolean; selIdx: number; filteredPhotos: any[];
  handleCreateFromSelection: () => void; handleGenerate: () => void;
}) {
  return (
    <>
                  {/* Base photo — always rendered; eye overlay crossfades on top.
                      On failure: degrade to the thumbnail, never a black stage. */}
                  {!loupePreviewFailed ? (
                    <img
                      key={sel.path}
                      src={photoUrl(sel.path) + (loupeRetry ? `&_r=${loupeRetry}` : '')}
                      alt=""
                      onError={() => setLoupePreviewFailed(true)}
                      onLoad={e => setPhotoNatDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
                      style={{ maxWidth:'100%', maxHeight:'100%', objectFit:'contain', display:'block', userSelect:'none',
                        boxShadow:'var(--shadow-2)',
                        animation:'fadeIn .35s cubic-bezier(.2,0,0,1)',
                        outline: selectedIds.has(selId ?? '') ? `3px solid ${T.mark}` : 'none',
                        outlineOffset:'-3px', transition:'outline .22s ease',
                      }}
                    />
                  ) : (
                    <div className="flex flex-col items-center gap-3" style={{ maxWidth:'100%', maxHeight:'100%' }}>
                      <Thumb key={`loupe-fallback-${sel.path}`} path={sel.path} eager
                        className="block max-h-full w-auto max-w-none object-contain"
                        style={{ maxWidth:'100%', maxHeight:'100%', objectFit:'contain', boxShadow:'var(--shadow-2)' }}/>
                      <span className="t-label">Full preview unavailable — showing thumbnail</span>
                      <Button size="sm" variant="quiet" icon={<RefreshCw size={11}/>}
                        onClick={() => setLoupeRetry(n => n + 1)}>
                        Retry full preview
                      </Button>
                    </div>
                  )}
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
                        background: showEyeOverlay ? T.raisedHover : T.glass,
                        border: `1px solid ${showEyeOverlay ? T.ink : T.lineStrong}`,
                        backdropFilter:'blur(16px) saturate(1.1)',
                        WebkitBackdropFilter:'blur(16px) saturate(1.1)',
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
                        background:T.glass, border:`1px solid ${T.lineStrong}`,
                        borderRadius:'var(--r-md)', backdropFilter:'blur(16px) saturate(1.1)',
                        WebkitBackdropFilter:'blur(16px) saturate(1.1)', boxShadow:'var(--shadow-2)',
                        pointerEvents:'none' }}>
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
                    style={{ position:'absolute', left:12, top:'50%', transform:'translateY(-50%)', width:34, height:34, borderRadius:'var(--r-round)', display:'flex', alignItems:'center', justifyContent:'center', background:T.glass, backdropFilter:'blur(16px) saturate(1.1)', WebkitBackdropFilter:'blur(16px) saturate(1.1)', color:hasPrev?T.ink:T.ink3, opacity:hasPrev?1:0, border:`1px solid ${T.lineStrong}`, boxShadow:'var(--shadow-2)', pointerEvents:hasPrev?'auto':'none', cursor:'pointer', fontSize:'var(--text-md)' }}>‹</button>
                  <button onClick={() => hasNext && setSelId(filteredPhotos[selIdx+1].id)} disabled={!hasNext}
                    style={{ position:'absolute', right:12, top:'50%', transform:'translateY(-50%)', width:34, height:34, borderRadius:'var(--r-round)', display:'flex', alignItems:'center', justifyContent:'center', background:T.glass, backdropFilter:'blur(16px) saturate(1.1)', WebkitBackdropFilter:'blur(16px) saturate(1.1)', color:hasNext?T.ink:T.ink3, opacity:hasNext?1:0, border:`1px solid ${T.lineStrong}`, boxShadow:'var(--shadow-2)', pointerEvents:hasNext?'auto':'none', cursor:'pointer', fontSize:'var(--text-md)' }}>›</button>
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
                      <span style={{ fontSize:'var(--text-sm)', fontWeight:500, color:T.ink }}>{selectedIds.size} selected</span>
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
  );
}
