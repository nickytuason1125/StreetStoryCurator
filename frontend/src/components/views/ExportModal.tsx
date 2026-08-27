import { useState } from 'react';
import { Download } from 'lucide-react';
import { Button } from '../ui/Button';
import { Modal } from '../ui/Modal';
import { Thumb } from '../photo/Thumb';
import { API, photoUrl } from '../../lib/api';

/* ── Export Modal ────────────────────────────────────────────────── */
/* Extracted verbatim from App.tsx during the views split. */
export function ExportModal({ photos, filterGrade, onClose }: { photos: any[]; filterGrade: string | null; onClose: () => void }) {
  const [xmpState, setXmpState] = useState<'idle'|'busy'|'done'|'error'>('idle');
  const [xmpCount, setXmpCount] = useState(0);
  const [zipState, setZipState] = useState<'idle'|'busy'|'error'>('idle');

  const handleDownload = (p: any) => {
    const a = document.createElement('a');
    a.href = photoUrl(p.path); a.download = p.path.split(/[\\/]/).pop() || 'photo.jpg';
    a.click();
  };

  /* One ZIP download instead of N staggered anchor-clicks — Chromium/WebView2
   * silently blocks automatic multi-downloads after the first. */
  const handleDownloadAll = async () => {
    setZipState('busy');
    try {
      const res = await fetch(`${API}/api/export/batch-zip`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paths: photos.map(p => p.path) }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      if (!data.zip) throw new Error('No archive returned');
      const a = document.createElement('a');
      a.href = photoUrl(data.zip);
      a.download = data.zip.split(/[\\/]/).pop() || 'photos.zip';
      a.click();
      setZipState('idle');
    } catch {
      // Fall back to per-photo downloads so the action still does *something*.
      setZipState('error');
      photos.forEach((p, i) => setTimeout(() => handleDownload(p), i * 300));
    }
  };

  const handleExportXmp = async () => {
    setXmpState('busy');
    try {
      const res = await fetch(`${API}/api/export/metadata`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ photos: photos.map(p => ({
          path: p.path, grade: p.grade, score: p.score,
          critique: p.critique, breakdown: p.breakdown, nima_score: p.nima_score,
        })) }),
      });
      const data = await res.json();
      setXmpCount(data.exported ?? 0);
      setXmpState('done');
    } catch {
      setXmpState('error');
    }
  };

  return (
    <Modal
      title="Export photos"
      subtitle={<><span className="t-num">{photos.length}</span> photo{photos.length !== 1 ? 's' : ''}{filterGrade ? ` · ${filterGrade} only` : ''}</>}
      onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="solid" onClick={handleDownloadAll} disabled={zipState === 'busy'} icon={<Download size={11}/>}>
            {zipState === 'busy' ? 'Zipping…' : <>Download all (<span className="t-num">{photos.length}</span>)</>}
          </Button>
        </>
      }
    >
      {photos.map(p => (
        <div key={p.id} className="flex items-center gap-3 border-b border-line py-2 last:border-0">
          <span className="relative block h-8 shrink-0 overflow-hidden rounded-sm bg-well">
            <Thumb path={p.path} className="block h-full w-auto max-w-none"/>
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm text-ink">{p.path.split(/[\\/]/).pop()}</p>
            <p className="t-num mt-px truncate text-xs text-ink-3">
              {[p.exif?.camera, p.exif?.aperture, p.exif?.shutter, p.exif?.iso ? `ISO ${p.exif.iso}` : null].filter(Boolean).join(' · ')}
            </p>
          </div>
          <Button size="sm" variant="quiet" onClick={() => handleDownload(p)}
            title={`Download ${p.path.split(/[\\/]/).pop()}`} icon={<Download size={10}/>}/>
        </div>
      ))}

      {/* XMP sidecars. State is carried by the words, not by a colour change —
          "Written" is unambiguous without turning the control green. */}
      <div className="sticky bottom-0 -mx-4 mt-2 flex items-center justify-between gap-3 border-t border-line bg-raised px-4 py-3">
        <div className="min-w-0">
          <p className="text-sm text-ink-2">XMP sidecars</p>
          <p className="mt-px text-sm text-ink-3">
            {xmpState === 'idle'  && 'Write .xmp files beside each photo — Lightroom and Capture One read them'}
            {xmpState === 'busy'  && 'Writing sidecars…'}
            {xmpState === 'done'  && <><span className="t-num">{xmpCount}</span> sidecar{xmpCount !== 1 ? 's' : ''} written beside your photos</>}
            {xmpState === 'error' && 'Export failed. Check the server log for the cause.'}
          </p>
        </div>
        <Button onClick={handleExportXmp} disabled={xmpState === 'busy'}
          variant={xmpState === 'error' ? 'danger' : 'solid'}>
          {xmpState === 'busy' ? 'Writing…' : xmpState === 'done' ? 'Written' : 'Export XMP'}
        </Button>
      </div>
    </Modal>
  );
}
