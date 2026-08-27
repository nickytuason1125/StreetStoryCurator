import { useEffect, useState } from 'react';
import { X, ArrowUp, FolderOpen, HardDrive } from 'lucide-react';
import { Button } from '../ui/Button';
import { cn } from '../../lib/cn';
import { Thumb } from '../photo/Thumb';
import { API } from '../../lib/api';

/* ── Folder browser modal ───────────────────────────────────────── */
/* Native-feeling explorer for picking the working folder. Extracted
 * verbatim from App.tsx during the views split; all state stays owned
 * by App and arrives through props. */
export function FolderBrowser({ mode, bPath, setBPath, bFolders, bImages, bSelFolders, setBSelFolders, loading, onNavigate, onGoUp, onFolderClick, onAddFolders, onUseFolder, onClose }: {
  mode: 'add' | 'use';
  bPath: string;
  setBPath: (p: string) => void;
  bFolders: string[];
  bImages: string[];
  bSelFolders: Set<string>;
  setBSelFolders: (s: Set<string>) => void;
  loading: boolean;
  onNavigate: (path: string) => void;
  onGoUp: () => void;
  onFolderClick: (e: React.MouseEvent, path: string, idx: number) => void;
  onAddFolders: (folders: string[]) => Promise<void>;
  onUseFolder: () => void;
  onClose: () => void;
}) {
  /* Roots come from the SERVER, which is the side that can actually see the
   * disk. These used to be five hardcoded literals pointing at one developer's
   * user profile, so on any other machine every shortcut was a dead link. And
   * no drive was listed at all — the browser only ever lists a directory you
   * have already named, so a library on D:/ or E:/ simply could not be reached
   * without typing the path by hand. /api/places answers both. */
  const [places, setPlaces] = useState<{ label: string; path: string }[]>([]);
  const [drives, setDrives] = useState<{ label: string; path: string }[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API}/api/places`)
      .then(r => (r.ok ? r.json() : null))
      .then(d => {
        if (cancelled || !d) return;
        setPlaces(d.places ?? []);
        setDrives(d.drives ?? []);
      })
      .catch(() => {/* typing a path still works; this is a convenience */});
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-scrim p-4">
      <div className="flex h-[82vh] w-full max-w-[640px] flex-col overflow-hidden rounded-md border border-line-strong bg-surface shadow-lg">
        <div className="flex shrink-0 items-center justify-between border-b border-line px-4 py-3">
          <span className="t-label">Choose a photo folder</span>
          <button onClick={onClose} aria-label="Close"
            className="cursor-pointer rounded-sm border-0 bg-transparent p-1 text-ink-3 transition-colors duration-fast ease hover:bg-raised hover:text-ink">
            <X size={16}/>
          </button>
        </div>

        <div className="flex shrink-0 items-center gap-2 border-b border-line bg-well px-3 py-2">
          <Button size="sm" onClick={onGoUp} icon={<ArrowUp size={11}/>}>Up</Button>
          <span className="t-num flex-1 truncate rounded-sm border border-line-strong bg-raised px-2 py-1 text-xs text-ink-2">
            {bPath}
          </span>
          <Button
            size="sm"
            variant="solid"
            onClick={async () => {
              try {
                if (mode === 'add') {
                  const toAdd = bSelFolders.size ? Array.from(bSelFolders) : [bPath];
                  await onAddFolders(toAdd);
                } else {
                  onUseFolder();
                }
              } catch (err) { /* non-blocking */ }
              onClose();
              setBSelFolders(new Set());
            }}
            disabled={bImages.length===0}
            title={bImages.length === 0 ? 'This folder holds no images' : undefined}>
            {mode === 'add' ? 'Add' : 'Use folder'}
            {bImages.length > 0 && <span className="t-num ml-1 opacity-70">{bImages.length}</span>}
          </Button>
        </div>
        <div className="flex flex-1 overflow-hidden">
          <div className="flex w-sidebar shrink-0 flex-col gap-px overflow-y-auto border-r border-line bg-well p-2">
            <p className="t-label mb-1 px-2">Quick access</p>
            {places.map(loc => (
              <button key={loc.path} onClick={() => { setBPath(loc.path); onNavigate(loc.path); }}
                className={cn(
                  'truncate rounded-sm border-0 px-2 py-1 text-left text-sm',
                  'cursor-pointer transition-colors duration-fast ease',
                  bPath === loc.path
                    ? 'bg-raised-hover text-ink'
                    : 'bg-transparent text-ink-3 hover:bg-raised hover:text-ink-2',
                )}>
                {loc.label}
              </button>
            ))}

            {/* Drives. Without these a library anywhere but the system drive
                was unreachable: browse-folder lists a directory you have
                already named, and nothing ever named D:/ or E:/. */}
            {drives.length > 0 && (
              <p className="t-label mb-1 mt-3 px-2">Drives</p>
            )}
            {drives.map(d => (
              <button key={d.path} onClick={() => { setBPath(d.path); onNavigate(d.path); }}
                title={d.path}
                className={cn(
                  'flex items-center gap-2 truncate rounded-sm border-0 px-2 py-1 text-left text-sm',
                  'cursor-pointer transition-colors duration-fast ease',
                  bPath === d.path
                    ? 'bg-raised-hover text-ink'
                    : 'bg-transparent text-ink-3 hover:bg-raised hover:text-ink-2',
                )}>
                <HardDrive size={12} className="shrink-0"/>
                <span className="truncate">{d.label}</span>
              </button>
            ))}
          </div>

            <div className="flex-1 overflow-y-auto p-4">
              {loading ? (
                <div className="flex h-full items-center justify-center text-ink-3">
                  <span className="text-sm">Reading folder…</span>
                </div>
              ) : bFolders.length===0 && bImages.length===0 ? (
                <div className="flex h-full flex-col items-center justify-center gap-2 text-ink-3">
                  <FolderOpen size={28} strokeWidth={1.5}/>
                  <p className="text-sm">Nothing here</p>
                  <p className="text-sm text-ink-3">No folders or images in this location.</p>
                </div>
              ) : (
                <>
                  {bFolders.length > 0 && (
                    <div className="mb-6">
                      <p className="t-label mb-2">Folders <span className="t-num">{bFolders.length}</span></p>
                      <div className="grid gap-1" style={{ gridTemplateColumns:'repeat(auto-fill, minmax(150px,1fr))' }}>
                        {bFolders.map((f, idx) => (
                          <button key={f} onClick={(e) => onFolderClick(e, f, idx)}
                            className={cn(
                              'flex cursor-pointer items-center gap-2 rounded-sm border px-3 py-2 text-left',
                              'transition-colors duration-fast ease',
                              bSelFolders.has(f)
                                ? 'border-mark bg-raised text-ink'
                                : 'border-line-strong bg-raised text-ink-2 hover:bg-raised-hover hover:text-ink',
                            )}>
                            <FolderOpen size={13} className="shrink-0 text-ink-3"/>
                            <span className="truncate text-sm">{f.split(/[\\/]/).pop()}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {bImages.length > 0 && (
                    <div>
                      <p className="t-label mb-2">Images <span className="t-num">{bImages.length}</span></p>
                      <div className="flex flex-wrap gap-1">
                        {bImages.slice(0,30).map(img => (
                          <div key={img} className="relative overflow-hidden rounded-sm border border-line bg-well">
                            <Thumb path={img} className="block h-thumb w-auto max-w-none"/>
                          </div>
                        ))}
                        {bImages.length > 30 && (
                          <div className="flex h-thumb items-center justify-center rounded-sm border border-line bg-raised px-3">
                            <span className="t-num text-xs text-ink-3">+{bImages.length-30} more</span>
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
  );
}
