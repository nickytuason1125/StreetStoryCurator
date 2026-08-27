import { useEffect, useRef } from 'react';
import { Download, ImageOff } from 'lucide-react';
import { Button } from '../ui/Button';
import { T, gradeRule, formatScore } from '../../theme/tokens';
import { cn } from '../../lib/cn';
import { Thumb } from '../photo/Thumb';

/* Scroll sentinel — asks for the next chunk of duplicates groups as it
 * approaches the viewport. IntersectionObserver, so no scroll handlers and
 * no work while the user isn't near the bottom. */
function RevealSentinel({ onVisible }: { onVisible: () => void }) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const io = new IntersectionObserver(
      es => { if (es.some(e => e.isIntersecting)) onVisible(); },
      { rootMargin: '600px' },
    );
    io.observe(node);
    return () => io.disconnect();
  }, [onVisible]);
  return <div ref={ref} className="h-8" aria-hidden />;
}

/* ── Similar Shots (duplicates) view ────────────────────────────── */
/* Group stats arrive from App's dupStats memo — the same source as the tab
 * count, so the two numbers can never disagree. Groups render incrementally:
 * `shownCount` at a time, extended by the sentinel. Extracted verbatim from
 * the App.tsx IIFE during the views split. */
export function SimilarShots({ groups, totalDups, shownCount, onRevealMore, onOpenPhoto, onExport }: {
  groups: any[];
  totalDups: number;
  shownCount: number;
  onRevealMore: () => void;
  onOpenPhoto: (id: string) => void;
  onExport: () => void;
}) {
  const shown = groups.slice(0, shownCount);

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden bg-ground">
      <div className="flex h-8 shrink-0 items-center gap-2 border-b border-line bg-surface px-4">
        <span className="t-label">Similar shots</span>
        <span className="text-xs text-ink-3">
          <span className="t-num">({groups.length})</span> group{groups.length!==1?'s':''}
          {' · '}<span className="t-num">({totalDups})</span> alternates
        </span>
        <div className="ml-auto">
          <Button size="sm" onClick={onExport} icon={<Download size={11}/>}>
            Export
          </Button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {shown.map((g, gi) => {
          const bestRule = gradeRule(g.best.grade);
          return (
            <div key={gi} className={cn(gi < shown.length - 1 && 'mb-6')}>

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
                          onClick={() => onOpenPhoto(g.best.id)}>
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
                              onClick={() => onOpenPhoto(p.id)}
                              className={cn(
                                'flex cursor-pointer flex-col border-0 bg-surface p-0',
                                'rounded-sm outline outline-2 outline-offset-1',
                                'transition-[outline-color] duration-fast ease',
                                isBest ? 'outline-ink' : 'outline-transparent hover:outline-line-strong',
                              )}>

                              <span className="relative block overflow-hidden bg-well" style={{ height: 116 }}>
                                <Thumb path={p.path}
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

                {shown.length < groups.length && (
                  <div>
                    <RevealSentinel onVisible={onRevealMore}/>
                    <p className="pb-3 text-center text-xs text-ink-3">
                      Showing <span className="t-num">{shown.length}</span> of <span className="t-num">{groups.length}</span> groups
                    </p>
                  </div>
                )}

                {groups.length === 0 && (
                  <div className="flex flex-col items-center justify-center gap-3 pt-12 text-ink-3">
                    <ImageOff size={28} strokeWidth={1}/>
                    <p className="text-sm">No similar shots found.</p>
                    <p className="max-w-[36ch] text-center text-sm text-ink-3">
                      Bursts and near-duplicates appear here once a folder has been graded.
                    </p>
                  </div>
                )}
      </div>
    </div>
  );
}
