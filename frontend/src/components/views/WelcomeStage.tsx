import { Aperture, FolderOpen } from 'lucide-react';
import { Button } from '../ui/Button';
import { T } from '../../theme/tokens';

/* ── WelcomeStage ─────────────────────────────────────────────────
 * The empty stage: blueprint grid, ambient glows, the brand card and
 * the resume strip. Extracted verbatim from App.tsx during the views
 * split — zero behaviour change, one file per view.
 *
 * The one screen with no photographs on it, so the mark colour may
 * breathe here without contaminating a read. */
export function WelcomeStage({ catalogBanner, onOpenFolder, onResume, onStartFresh }: {
  catalogBanner: boolean; onOpenFolder: () => void; onResume: () => void; onStartFresh: () => void;
}) {
  return (
    <div className="relative flex w-full items-center justify-center">
                  {/* Blueprint grid — fine engineering lines under everything,
                      fading out radially so it reads as depth, not wallpaper. */}
                  <div aria-hidden className="pointer-events-none absolute inset-0"
                    style={{
                      backgroundImage: `linear-gradient(var(--line) 1px, transparent 1px),
                        linear-gradient(90deg, var(--line) 1px, transparent 1px)`,
                      backgroundSize: '36px 36px',
                      maskImage: 'radial-gradient(circle at 50% 45%, black 0%, transparent 68%)',
                      WebkitMaskImage: 'radial-gradient(circle at 50% 45%, black 0%, transparent 68%)',
                      opacity: .45 }}/>
                  {/* Ambient backdrop — two soft glows, warm over cool. This is
                      the one screen with no photographs on it, so the mark
                      colour may breathe here without contaminating a read. */}
                  <div aria-hidden className="pointer-events-none absolute"
                    style={{ width: 760, height: 760, borderRadius: 'var(--r-round)',
                      background: `radial-gradient(circle, ${T.markDim} 0%, transparent 60%)`,
                      filter: 'blur(48px)', opacity: .8 }}/>
                  <div aria-hidden className="pointer-events-none absolute"
                    style={{ width: 900, height: 900, borderRadius: 'var(--r-round)',
                      background: `radial-gradient(circle, ${T.raised} 0%, transparent 55%)`,
                      filter: 'blur(64px)', opacity: .5 }}/>
                  {/* Live gradient — a conic aurora drifting slowly behind the card.
                      Stops are token-only (mark-dim / raised / a whisper of ink) so
                      the light stays hue-safe; 26 s per revolution reads as ambient
                      weather, not a spinner. */}
                  <div aria-hidden className="pointer-events-none absolute"
                    style={{ width: 1100, height: 1100, borderRadius: 'var(--r-round)',
                      background: `conic-gradient(from 0deg,
                        transparent 0deg,
                        ${T.markDim} 55deg,
                        transparent 125deg,
                        ${T.raised} 195deg,
                        transparent 265deg,
                        color-mix(in srgb, ${T.ink} 6%, transparent) 320deg,
                        transparent 360deg)`,
                      filter: 'blur(72px)', opacity: .55,
                      animation: 'auroraDrift 26s linear infinite' }}/>
                  <div className="relative elev-3 w-full rounded-md border border-line bg-surface"
                    style={{ maxWidth: 'var(--w-hero)', animation: 'dialogIn .5s var(--ease)' }}>
                  {/* Gradient hairline — one light seam along the card's top edge,
                      brightest at centre. Gives the flat surface a machined edge. */}
                  <div aria-hidden className="pointer-events-none absolute inset-x-6 top-0 h-px"
                    style={{ background: `linear-gradient(90deg, transparent 0%,
                      color-mix(in srgb, ${T.ink} 30%, transparent) 50%, transparent 100%)` }}/>
                  {/* Scan sweep — a faint band crossing the card like a sensor
                      reading a frame. Clipped to the card's rounded corners. */}
                  <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden rounded-md">
                    <div style={{ position: 'absolute', left: 0, right: 0, top: 0, height: 72,
                      background: `linear-gradient(180deg, transparent,
                        color-mix(in srgb, ${T.ink} 5%, transparent), transparent)`,
                      opacity: .6, animation: 'scanSweep 7s linear infinite' }}/>
                  </div>
                  {/* Viewfinder brackets — HUD reticle ticks at the four corners. */}
                  {([
                    { top: -1, left: -1 }, { top: -1, right: -1 },
                    { bottom: -1, left: -1 }, { bottom: -1, right: -1 },
                  ] as Array<{ top?: number; right?: number; bottom?: number; left?: number }>).map((c, i) => {
                    const edge = '1px solid var(--line-strong)';
                    return (
                      <div key={i} aria-hidden className="pointer-events-none absolute"
                        style={{ width: 14, height: 14,
                          ...(c.top !== undefined ? { top: c.top } : {}),
                          ...(c.right !== undefined ? { right: c.right } : {}),
                          ...(c.bottom !== undefined ? { bottom: c.bottom } : {}),
                          ...(c.left !== undefined ? { left: c.left } : {}),
                          borderTop:    c.top !== undefined    ? edge : undefined,
                          borderRight:  c.right !== undefined  ? edge : undefined,
                          borderBottom: c.bottom !== undefined ? edge : undefined,
                          borderLeft:   c.left !== undefined   ? edge : undefined }}/>
                    );
                  })}
                  <div className="flex flex-col gap-6 px-8 py-8">

                    {/* Brand row: aperture mark + engine label, with a live-
                        status readout on the right — mono, tabular, pulsing. */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Aperture size={16} strokeWidth={2} style={{ color: T.mark }}/>
                        <span className="t-label">Photo culling engine</span>
                      </div>
                      <span className="t-num flex items-center gap-2 text-xs text-ink-3">
                        <span aria-hidden className="rounded-full"
                          style={{ width: 6, height: 6, background: T.mark,
                            animation: 'pulseDot 2.4s ease-in-out infinite' }}/>
                        READY
                      </span>
                    </div>

                    {/* Headline — the mark-coloured full stop is the signature:
                        small, confident, the only warm glyph on the stage. */}
                    <div className="flex flex-col gap-2">
                      <h1 className="t-display text-ink">
                        FrameGrade<span style={{ color: T.mark }}>.</span>
                      </h1>
                      <p className="text-md text-ink-2" style={{ maxWidth: '46ch' }}>
                        Open a folder of photos and get straight to your
                        best shots.
                      </p>
                    </div>

                    {/* CTA */}
                    <div className="flex items-center gap-3">
                      <Button variant="ink" size="md" onClick={onOpenFolder}
                        icon={<FolderOpen size={14} strokeWidth={1.5}/>}>
                        Open folder
                      </Button>
                    </div>
                  </div>

                  {/* Resume strip */}
                  <div className="flex items-center gap-3 border-t border-line bg-ground px-8 py-3">
                    {catalogBanner ? (
                      <>
                        <span className="flex-1 text-sm text-ink-3">Pick up where you left off?</span>
                        <Button variant="default" size="sm" onClick={onResume}>Resume</Button>
                        <Button size="sm" variant="quiet" onClick={onStartFresh}>
                          Start fresh
                        </Button>
                      </>
                    ) : (
                      <span className="text-sm text-ink-3">
                        50 to 100 photos is a good first run
                      </span>
                    )}
                  </div>
                  </div>
                </div>
  );
}