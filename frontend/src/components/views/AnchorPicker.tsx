import { T } from '../../theme/tokens';
import { Thumb } from '../photo/Thumb';
import { useWindowedGrid } from '../../hooks/useWindowedGrid';

/* ── AnchorPicker ─────────────────────────────────────────────────
 * The Creative Director's "click a photo to set the anchor" grid.
 *
 * Windowed with the same hook as the contact sheet: this grid used to
 * live inside the config panel's scroll region and mounted one eager
 * <img> per photo in the library — 21,416 DOM image nodes on the live
 * catalog, re-reconciled on every state change. It now owns a fixed
 * scroll viewport and renders only the rows near it (±3), with spacer
 * padding preserving the scroll height.
 *
 * Cells are verbatim from the pre-extraction picker: 3:2, object-cover
 * (the anchor preview is a crop by design — the full frame is judged in
 * the loupe), score overlaid on a scrim gradient, anchor state via
 * outline + scale. */

export function AnchorPicker({ photos, anchorPath, onPick }: {
  photos: any[]; anchorPath: string | null; onPick: (path: string | null) => void;
}) {
  /* rowExtra = row gap only — the score strip overlays the photo, so a
   * row is exactly one 3:2 cell tall plus the 4px grid gap. */
  const wg = useWindowedGrid({ itemCount: photos.length, minColWidth: 96, gap: 4, rowExtra: 4 });
  return (
    <div ref={wg.ref} onScroll={wg.onScroll} className="min-h-0 flex-1 overflow-y-auto rounded-sm bg-ground">
      <div className="grid gap-1"
        style={{ gridTemplateColumns: `repeat(${wg.cols}, minmax(0, 1fr))`,
                 paddingTop: wg.padTop, paddingBottom: wg.padBottom }}>
        {photos.slice(wg.first, wg.last).map(p => {
          const isAnchor = p.path === anchorPath;
          return (
            <button key={p.id} onClick={() => onPick(isAnchor ? null : p.path)}
              style={{ position:'relative', aspectRatio:'3/2', padding:0, border:'none', borderRadius:'var(--r-sm)', overflow:'hidden', cursor:'pointer',
                outline: isAnchor ? `2px solid ${T.mark}` : `1px solid ${T.line}`, outlineOffset: isAnchor ? 2 : 0,
                transform: isAnchor ? 'scale(1.05)' : 'scale(1)', transition:'transform .12s, outline .12s' }}>
              <Thumb path={p.path} eager style={{ width:'100%', height:'100%', objectFit:'cover', display:'block' }}/>
              <div style={{ position:'absolute', bottom:0, left:0, right:0, height:14, background:`linear-gradient(transparent, ${T.scrim})`, display:'flex', alignItems:'center', justifyContent:'flex-end', padding:'0 4px' }}>
                <span className="t-num text-xs text-ink opacity-80">{Math.round(p.score*100)}</span>
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
  );
}