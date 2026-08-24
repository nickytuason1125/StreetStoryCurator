import { Wand2, Download, RefreshCw, Layers, Upload, X } from 'lucide-react';
import { Button } from '../ui/Button';
import { Field, TextArea } from '../ui/Field';
import { Segmented } from '../ui/Segmented';
import { Thumb } from '../photo/Thumb';
import { AnchorPicker } from './AnchorPicker';
import { T, gradeLabel } from '../../theme/tokens';
import { API, photoUrl } from '../../lib/api';
import { gc } from '../../lib/grading';
import { cn } from '../../lib/cn';

/* â”€â”€ CreativeDirector â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
 * The whole Creative Direction view: the config sidebar (mood brief,
 * reference PDFs, anchor picker, build controls) and the sequence
 * results grid. Extracted verbatim from App.tsx during the views
 * split â€” props carry the same names as the App state they wrap. */
export function CreativeDirector({
  photos, creativeResults, creativeLoading, engineHealth,
  creativePrompt, setCreativePrompt,
  ragPdfs, ragUploading, handleRagUpload, handleRagClear,
  creativeAnchor, setCreativeAnchor, seqMode, setSeqMode,
  handleRunCreativeDirection, handleSaveSequence, sequenceSaving,
  creativeSelection, creativeFallback, creativeProgress, creativeStage,
  creativeCount, setCreativeCount, usedCount, handleClearUsed,
  pegFile, setPegFile, pegHash, setPegHash, pegLoading, handlePegUpload,
}: {
  photos: any[]; creativeResults: any[]; creativeLoading: boolean; engineHealth: any;
  creativePrompt: string; setCreativePrompt: (v: string) => void;
  ragPdfs: any[]; ragUploading: boolean; handleRagUpload: any; handleRagClear: () => void;
  creativeAnchor: string | null; setCreativeAnchor: (v: string | null) => void;
  seqMode: string; setSeqMode: (v: any) => void;
  handleRunCreativeDirection: () => void; handleSaveSequence: () => void; sequenceSaving: boolean;
  creativeSelection: any; creativeFallback: string | null; creativeProgress: any; creativeStage: string | null;
  creativeCount: any; setCreativeCount: (v: any) => void; usedCount: any; handleClearUsed: () => void;
  pegFile: any; setPegFile: (v: any) => void; pegHash: any; setPegHash: (v: any) => void;
  pegLoading: boolean; handlePegUpload: any;
}) {
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
                  <span className="t-label">Creative director</span>
                </div>
                <p className="text-xs text-ink-3">
                  Builds a story arc from five visually distinct frames.
                </p>
              </div>

              <div className="flex flex-col gap-6 overflow-y-auto px-4 py-4"
                style={sortedPhotos.length > 0 ? { flex:'0 1 auto', minHeight:0 } : undefined}>

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
                  <div className="flex items-center gap-3">
                    <input type="range" min={4} max={10} step={1} value={creativeCount}
                      onChange={e => setCreativeCount(Number(e.target.value))}
                      aria-label="Sequence length"
                      aria-valuetext={`${creativeCount} photos`}
                      className="range-token flex-1" />
                    <span className="shrink-0 text-right text-sm text-ink-2 tabular-nums">
                      {creativeCount} photos
                    </span>
                  </div>
                </div>

                {/* Reference photo */}
                <div>
                  <label className="t-label mb-1 block">
                    Reference Photo{' '}
                    <span className="text-xs font-normal normal-case tracking-normal text-ink-3">optional</span>
                  </label>
                  <p className="mb-3 text-xs text-ink-3">Sets the visual style anchor for the sequence.</p>
                  {creativeAnchor ? (
                    <div style={{ position:'relative', borderRadius:'var(--r-md)', overflow:'hidden', border:`2px solid ${T.mark}`, cursor:'pointer', boxShadow:`0 0 0 3px ${T.markDim}` }}
                      onClick={()=>setCreativeAnchor(null)} title="Click to remove">
                      <Thumb path={creativeAnchor} eager style={{ width:'100%', aspectRatio:'3/2', objectFit:'cover', display:'block' }}/>
                      <div className="t-label absolute left-1 top-1 rounded-sm px-1" style={{ background:T.mark, color:T.well }}>ANCHOR</div>
                      <div style={{ position:'absolute', top:6, right:6, background:T.scrim, backdropFilter:'blur(4px)', borderRadius:'var(--r-sm)', padding:'3px 8px', fontSize:'var(--text-xs)', color:T.ink, fontWeight:600 }}>✕ remove</div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center gap-2 rounded-md border border-line-strong bg-well py-6 text-sm text-ink-3">
                      <Wand2 size={14} strokeWidth={1.5}/>
                      <span>Click a photo below to set anchor</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Photo picker — own scroll viewport, windowed rows. Extracted to
                  AnchorPicker: it used to sit inside the panel's scroll region and
                  mount one eager <img> per library photo (21,416 on the live
                  catalog). Now only the visible rows exist. */}
              {sortedPhotos.length > 0 && (
                <div className="flex min-h-0 flex-1 flex-col px-4 pb-4">
                  <p className="mb-2 text-xs text-ink-3">{sortedPhotos.length} photos · sorted by grade · click to anchor</p>
                  <AnchorPicker photos={sortedPhotos} anchorPath={creativeAnchor} onPick={setCreativeAnchor}/>
                </div>
              )}

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
                      <span style={{fontSize:'var(--text-sm)', fontWeight:500}}>Story sequence</span>
                      <span style={{fontSize:'var(--text-xs)', color:T.ink3, background:T.raised, borderRadius:'var(--r-sm)', padding:'2px 8px'}}>{successResults.length} images</span>
                      {creativeResults.some((r:any)=>!r.success) && (
                        <span style={{fontSize:'var(--text-xs)', color:T.gradeWeak, cursor:'default'}}
                          title={creativeResults.filter((r:any)=>!r.success).map((r:any)=>`${(r.source_path??'').split(/[\\/]/).pop()}: ${r.error??'failed'}`).join('\n')}>
                          {creativeResults.filter((r:any)=>!r.success).length} failed ⓘ
                        </span>
                      )}
                      {creativeSelection?.cohesion_mean != null && (
                        <span style={{fontSize:'var(--text-xs)', color:T.ink3, cursor:'default'}}
                          title={`Chosen from ${creativeSelection.pool_size ?? '?'} graded photos. `
                            + `Cohesion ${Number(creativeSelection.cohesion_mean).toFixed(2)} `
                            + `(lowest ${Number(creativeSelection.cohesion_min ?? 0).toFixed(2)}). `
                            + `Higher means the set hangs together more tightly; `
                            + `too high and it is repetitive.`}>
                          cohesion {Number(creativeSelection.cohesion_mean).toFixed(2)} ⓘ
                        </span>
                      )}
                      {creativeSelection?.reason && (
                        <span style={{fontSize:'var(--text-xs)', color:T.gradeWeak, cursor:'default'}}
                          title={String(creativeSelection.reason)}>
                          fewer than asked ⓘ
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
                              <span style={{fontSize:'var(--text-xs)', fontWeight:600, letterSpacing:'.12em', color:sc, textTransform:'uppercase', flex:1}}>{slot}</span>
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
                                  <span style={{fontSize:'var(--text-sm)', color:T.ink, fontVariantNumeric:'tabular-nums'}}>{Math.round(photoScore*100)}</span>
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
                <div style={{flex:1, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:14, color:T.ink3, padding:40}}>
                  <Wand2 size={26} strokeWidth={1} style={{opacity:.25}}/>
                  <div style={{textAlign:'center', maxWidth:340}}>
                    <p className="t-label" style={{margin:'0 0 8px'}}>No sequence yet</p>
                    <p style={{fontSize:'var(--text-sm)', lineHeight:1.7, margin:0, color:T.ink3}}>
                      Write a mood brief on the left, optionally pick a reference photo, then press Build the story.
                    </p>
                    {photos.length===0 && (
                      <p style={{fontSize:'var(--text-sm)', color:T.gradeWeak, marginTop:12}}>Grade a folder first to load photos.</p>
                    )}
                  </div>
                </div>
              )}

            </div>
          </div>
          );
}

