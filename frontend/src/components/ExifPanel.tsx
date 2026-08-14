import { cn } from '../lib/cn';

/* The EXIF tab.
 *
 * Lives here rather than in App.tsx so it sits in the STRICT lint zone, where
 * literal values are held to zero. The version it replaces was the last thing
 * in the app still styled with inline hex and `fontFamily: 'SF Mono'` — a
 * macOS-only font that silently fell back to generic monospace on this machine,
 * so the one panel that is nothing but numbers was the one panel not set in the
 * app's actual mono face.
 *
 * Grouped rather than one flat list of 25 rows: a photographer looks for one of
 * four things — what shot it, how it was exposed, what the frame is, and when.
 * Keys are label-cased, values are tabular mono so the digits align down the
 * column and two frames can be compared by eye.
 */

type Row = readonly [key: string, label: string];

const GROUPS: readonly (readonly [string, readonly Row[]])[] = [
  ['Camera', [
    ['camera', 'Body'],
    ['lens', 'Lens'],
    ['firmware', 'Firmware'],
    ['body_serial', 'Body serial'],
    ['lens_serial', 'Lens serial'],
  ]],
  ['Exposure', [
    ['aperture', 'Aperture'],
    ['shutter', 'Shutter'],
    ['iso', 'ISO'],
    ['ev', 'Exp. bias'],
    ['program', 'Mode'],
    ['metering', 'Metering'],
    ['white_balance', 'White balance'],
    ['flash', 'Flash'],
  ]],
  ['Frame', [
    ['focal', 'Focal length'],
    ['focal_35mm', '35mm equiv.'],
    ['subject_distance', 'Subject distance'],
    ['dimensions', 'Dimensions'],
    ['megapixels', 'Resolution'],
    ['orientation', 'Orientation'],
    ['color_space', 'Colour space'],
    ['format', 'Format'],
    ['file_size', 'File size'],
  ]],
  ['Time', [
    ['date', 'Date'],
    ['time', 'Time'],
  ]],
  ['Location', [
    ['gps', 'Coordinates'],
  ]],
  ['Rights', [
    ['artist', 'Artist'],
    ['copyright', 'Copyright'],
  ]],
] as const;

export function ExifPanel({ exif }: { exif: Record<string, unknown> | null | undefined }) {
  const present = (k: string) => {
    const v = exif?.[k];
    return v != null && String(v).length > 0;
  };

  const groups = GROUPS
    .map(([title, rows]) => [title, rows.filter(([k]) => present(k))] as const)
    .filter(([, rows]) => rows.length > 0);

  if (!groups.length) {
    return (
      <p className="text-sm text-ink-3">
        No EXIF recorded for this photo.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {groups.map(([title, rows]) => (
        <section key={title}>
          <h3 className="t-label mb-1">{title}</h3>
          <dl className="m-0 flex flex-col">
            {rows.map(([k, label], i) => (
              <div
                key={k}
                className={cn(
                  'flex items-baseline justify-between gap-3 py-1',
                  i > 0 && 'border-t border-line',
                )}
              >
                <dt className="shrink-0 text-xs text-ink-3">{label}</dt>
                <dd className="t-num m-0 min-w-0 break-words text-right text-xs text-ink-2">
                  {String(exif?.[k])}
                </dd>
              </div>
            ))}
          </dl>
        </section>
      ))}
    </div>
  );
}
