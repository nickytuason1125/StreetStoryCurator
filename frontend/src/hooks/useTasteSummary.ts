import { useEffect, useState } from 'react';
import { API } from '../lib/api';

/* useTasteSummary — the loupe's taste-authority meter data.
 *
 * One tiny local GET (/api/taste/summary), module-cached for 30s so the
 * filmstrip, the HUD and the rail can all read it without hammering the
 * backend. `dep` (usually the selected photo's star count) forces a refresh
 * past the TTL the moment the user rates something — the meter must move
 * with the baseline, not lag a session behind it. */
export type TasteSummary = {
  ratings: number;
  weight: number;      // ceiling the blend actually uses right now
  next_at: number | null;
  next_weight: number | null;
};

let _cache: TasteSummary | null = null;
let _fetchedAt = 0;
const _TTL = 30_000;

export function useTasteSummary(dep?: unknown): TasteSummary | null {
  const [taste, setTaste] = useState<TasteSummary | null>(_cache);

  useEffect(() => {
    let live = true;
    if (_cache && Date.now() - _fetchedAt < _TTL) return;
    fetch(`${API}/api/taste/summary`)
      .then(r => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d: TasteSummary) => { _cache = d; _fetchedAt = Date.now(); if (live) setTaste(d); })
      .catch(() => { /* meter simply stays hidden — never block the loupe on it */ });
    return () => { live = false; };
  }, [dep]);

  return taste;
}