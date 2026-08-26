import { useCallback, useEffect, useRef, useState } from 'react';

/* useWindowedGrid — zero-dependency row windowing for the contact sheets.
 *
 * Why: GridView used to mount one <img> per photo in the library — 21,416 DOM
 * image nodes on the live catalog, all reconciled on every state change.
 * `loading="lazy"` defers the fetches but not the nodes, the layout, or the
 * reconciliation. This hook renders only the rows near the viewport (± overscan)
 * and spacers that preserve the scroll height, so a 100k library costs the
 * same as a 500-frame one.
 *
 * Contract: the caller renders `items.slice(first, last)` inside a CSS grid of
 * exactly `cols` columns, with paddingTop/Bottom from the hook, inside the
 * scroll container the hook measures. Cells are 3:2 (the contact sheet's
 * aspect) — row height is derived from the measured column width.
 *
 * Measurement is a ResizeObserver on the scroll container itself, so window
 * resizes and panel drags re-column the sheet for free.
 */

export interface WindowedGrid {
  /** Attach to the scroll container (the overflow-auto element). */
  ref: (node: HTMLDivElement | null) => void;
  /** Attach onScroll to the same element. */
  onScroll: (e: { currentTarget: { scrollTop: number } }) => void;
  /** Column count — render the grid with `repeat(cols, minmax(0, 1fr))`. */
  cols: number;
  /** First visible item index (inclusive). */
  first: number;
  /** One past the last visible item index. */
  last: number;
  /** Top spacer height in px. */
  padTop: number;
  /** Bottom spacer height in px. */
  padBottom: number;
  /** Computed pixel width of one column - lets callers cap/centre cells. */
  colWidth: number;
}

export function useWindowedGrid(opts: {
  itemCount: number;
  /** Minimum column width — matches the sheet's minmax() rhythm. */
  minColWidth: number;
  /** Optional hard cap on column count - few photos form a centred block. */
  maxCols?: number;
  /** Grid gap in px. */
  gap: number;
  /** Per-row height beyond the 3:2 image cell: rule + label strip + row gap. */
  rowExtra: number;
  overscanRows?: number;
}): WindowedGrid {
  const { itemCount, minColWidth, gap, rowExtra, overscanRows = 3 } = opts;
  const nodeRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);
  const [scrollTop, setScrollTop] = useState(0);
  const rafRef = useRef(0);

  useEffect(() => () => {
    // Unmount cleanup: a pending frame must not setState after teardown.
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
  }, []);

  const ref = useCallback((node: HTMLDivElement | null) => {
    nodeRef.current = node;
  }, []);

  useEffect(() => {
    const node = nodeRef.current;
    if (!node) return;
    const ro = new ResizeObserver(entries => {
      for (const e of entries) {
        setWidth(e.contentRect.width);
        setHeight(e.contentRect.height);
      }
    });
    ro.observe(node);
    return () => ro.disconnect();
  }, []);

  // With few photos a full-width single row reads as one stretched strip over
  // a void - aim for a near-square centred block instead; fitCols is the ceiling.
  const fitCols = Math.max(1, Math.floor((width + gap) / (minColWidth + gap)));
  const idealCols = Math.max(1, Math.round(Math.sqrt(itemCount * 2)));
  const cols = Math.min(fitCols, Math.max(1, opts.maxCols ?? idealCols));
  const colWidth = width > 0 ? (width - (cols - 1) * gap) / cols : minColWidth;
  const rowHeight = colWidth / 1.5 + rowExtra;
  const totalRows = Math.ceil(itemCount / cols);
  const startRow = Math.max(0, Math.floor(scrollTop / rowHeight) - overscanRows);
  const visibleRows = Math.ceil(height / Math.max(1, rowHeight)) + overscanRows * 2;
  const endRow = Math.min(totalRows, startRow + visibleRows);

  const onScroll = useCallback((e: { currentTarget: { scrollTop: number } }) => {
    // Scroll events fire many times per display frame; without coalescing each
    // one triggers a full grid re-render. One render per frame is the ceiling
    // the eye can perceive — everything between is dropped CPU work.
    const st = e.currentTarget.scrollTop;
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = 0;
      setScrollTop(st);
    });
  }, []);

  return {
    ref,
    onScroll,
    cols,
    colWidth,
    first: startRow * cols,
    last: endRow * cols,
    padTop: startRow * rowHeight,
    padBottom: Math.max(0, totalRows - endRow) * rowHeight,
  };
}
