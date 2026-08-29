import { useState, useEffect, useCallback, useMemo, useRef, memo } from "react";
import axios from "axios";
import {
  DndContext, closestCenter, KeyboardSensor, PointerSensor,
  useSensor, useSensors,
} from "@dnd-kit/core";
import {
  SortableContext, sortableKeyboardCoordinates,
  verticalListSortingStrategy, useSortable,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
  FolderOpen, Layers, FileDown, RefreshCw,
  ImageOff, X, Sparkles, Flag,
  LayoutGrid, RectangleHorizontal, SlidersHorizontal,
  Download, CheckSquare, ArrowUpDown, ArrowUp, ArrowDown,
  Wand2, Zap, Eye, EyeOff, Upload, Search, Aperture,
} from "lucide-react";
import { Button } from "./components/ui/Button";
import { Chip } from "./components/ui/Chip";
import { Segmented } from "./components/ui/Segmented";
import { AnnotatedMark } from "./components/ui/GradeRule";
import { Modal } from "./components/ui/Modal";
import { CommandPalette } from "./components/ui/CommandPalette";
import { KbdHint } from "./components/ui/Kbd";
import { Field, TextArea } from "./components/ui/Field";
import { StarRating } from "./components/ui/StarRating";
import { ExifPanel } from "./components/ExifPanel";
import { Thumb } from "./components/photo/Thumb";
import { Filmstrip, GridView } from "./components/views/GridView";
import { AnchorPicker } from "./components/views/AnchorPicker";
import { WelcomeStage } from "./components/views/WelcomeStage";
import { LoupeStage } from "./components/views/LoupeStage";
import { AnalysisPanel } from "./components/views/AnalysisPanel";
import { CreativeDirector } from "./components/views/CreativeDirector";
import { regionGuide, tierColor, tierIcon, tierHeat } from "./lib/regions";
import type { RegionTier } from "./lib/regions";
import { aspectDim } from "./lib/aspects";
import { ExportModal } from "./components/views/ExportModal";
import { SimilarShots } from "./components/views/SimilarShots";
import { FolderBrowser } from "./components/views/FolderBrowser";
import { T, gradeRule, gradeKey, gradeLabel, formatScore } from "./theme/tokens";
import { cn } from "./lib/cn";
import ErrorBoundary from "./ErrorBoundary";
import { API, photoUrl, sanitizePath, thumbUrl } from "./lib/api";
import { APP_VERSION } from "./lib/version";
import { useGuardedInterval } from "./hooks/useGuardedInterval";
import { useWindowedGrid } from "./hooks/useWindowedGrid";
import { gc, ramReadiness } from "./lib/grading";

/* The second palette that used to live here is gone.
 *
 * `C` declared its own twenty literals — a blue-black ground (#0a0a0d), a
 * blue-grey ink ramp and a green/amber/red grade set — while src/theme/tokens.css
 * declared a neutral ground (#1a1a1a), a neutral ink ramp and a teal/silent/dim
 * grade set. Both were live at once, in the same window, so the app rendered two
 * different greys depending on which half of the file drew the pixel, and editing
 * tokens.css moved only half the UI.
 *
 * Everything now reads T.* from theme/tokens.ts, which returns var(--x) rather
 * than a value, so tokens.css is the only file where a colour is decided. */

/* The grade vocabulary. Everything downstream inherits from these, which is why
 * they are the first thing converted — see src/theme/tokens.ts.
 *
 * Mid is deliberately silent. It is the majority bucket, so an amber badge on
 * every Mid frame put colour on most of the grid and made the machine's least
 * confident verdict its loudest. Neutral ink and no rule instead. */

/* Stable photo identity. IDs used to be `p-${index}`, so re-grading or
 * restoring the catalog re-indexed every photo mid-session — any selection
 * state keyed by id pointed at the wrong frame afterwards. Deriving the id
 * from the path keeps identity stable across reloads, re-grades and merges. */
const photoId = (path: string): string => {
  let h = 5381;
  for (let i = 0; i < path.length; i++) h = ((h << 5) + h + path.charCodeAt(i)) | 0;
  return `p-${(h >>> 0).toString(36)}`;
};

/* Vision-critique region guide + tier colours moved to lib/regions.ts —
 * shared by the loupe stage, the analysis panel and the critique heatmap. */

/* gLow() mapped each grade to a 14% tinted badge background. Deleted: the
 * machine's verdict is a 2px rule under the frame, not a filled badge behind
 * text. gl() was a duplicate of gradeLabel() in theme/tokens.ts. */
// gIcon() lived here — it mapped grades to ✅ / ⚠️ / ❌. It had no callers left,
// and emoji-as-status is the clearest "generated interface" tell there is. The
// grade is carried by the rule under each frame instead. Do not reintroduce it.

const _SLOGANS: Array<[RegExp, string]> = [
  // Patterns match the backend's (model-agnostic) progress wording. Never put a
  // model name in the SLOGAN text — these strings are shown to the user.
  [/scanning folder|found \d+|already graded/i, "Pulling the contact sheet…"],
  [/checking image files|unusable images/i,     "Culling the camera-shake casualties…"],
  [/analyz|image analysis/i,                    "Reading the light in every frame…"],
  [/near-duplicate|marking duplicates/i,        "Picking the best frame from each burst…"],
  [/preparing deep analysis/i,                  "The photo editor is pulling up a chair…"],
  [/judging each photo|deep analysis ready/i,   "Studying composition, moment, and story…"],
  [/scoring image quality|quality scoring/i,    "Running the darkroom technical check…"],
  [/light and contrast/i,                       "Measuring the exposure…"],
  [/style reference|creative brief/i,           "Comparing against the reference portfolio…"],
  [/taste profile/i,                            "Recalling your editorial eye…"],
  [/refining composition/i,                     "Second shooter weighing in…"],
  [/sequenc|assigning grades|building your gallery/i, "Building the selects…"],
  [/combining scores/i,                         "Matching each frame to its genre…"],
  [/saving results|photo details/i,             "Filing the contact sheet…"],
];
function toSlogan(desc: string): string {
  if (!desc) return '';
  for (const [re, slogan] of _SLOGANS) {
    if (re.test(desc)) return slogan;
  }
  return desc;
}

function SortableItem({ id, children }: { id: string; children: React.ReactNode }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id });
  const style = transform
    ? { transform: CSS.Transform.toString(transform), transition, opacity: isDragging ? 0.5 : 1, zIndex: isDragging ? 10 : 1 }
    : {};
  return <div ref={setNodeRef} style={style} {...attributes} {...listeners}>{children}</div>;
}

/* RevealSentinel moved to components/views/SimilarShots.tsx with its only
 * consumer. FilmThumb and GridView moved to components/views/GridView.tsx —
 * see that file. StarRating moved to components/ui/StarRating.tsx. The copy
 * that lived here shadowed it, so the shared component sat imported by nobody
 * while this one painted stars in oklch(70% .18 72) — an amber about ten
 * degrees in hue from the old grade accent. At 11px over a photograph the two
 * were the same colour, carrying two unrelated meanings inside one cell.
 * Stars are the photographer's judgement, so they belong to --mark; the
 * machine's grade gives up colour. */

/* ExifBlock moved to components/ExifPanel.tsx — see that file for why it is
 * grouped now. It was the last component still setting its values in
 * 'SF Mono', a macOS font that does not exist on this machine, so the one
 * panel that is entirely numbers was the one not set in the app's mono face. */

/* ExportModal moved to components/views/ExportModal.tsx. */

/* GridView moved to components/views/GridView.tsx. */

/* ── Critique trigger parser ────────────────────────────────────── */
// Parses <trigger type="blur|heatmap|grid">text</trigger> tags emitted by the
// jury LLM into hoverable inline spans that drive the image overlay state.
// Falls back to plain text if the LLM produces no tags.
function parseCritique(
  text: string,
  onEnter: (type: string) => void,
  onLeave: () => void,
): React.ReactNode[] {
  const RE = /<trigger\s+type="([^"]+)">([^<]*)<\/trigger>/g;
  const nodes: React.ReactNode[] = [];
  let last = 0;
  let key  = 0;
  let m: RegExpExecArray | null;
  const COLORS: Record<string, string> = {
    blur:    T.alarmWarn,
    heatmap: T.alarmCrit,
    grid:    T.ink2,
  };
  while ((m = RE.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const typ   = m[1];
    const label = m[2];
    const col   = COLORS[typ] ?? T.ink2;
    nodes.push(
      <span key={key++}
        onMouseEnter={() => onEnter(typ)}
        onMouseLeave={onLeave}
        style={{ cursor:'pointer', textDecorationLine:'underline', textDecorationStyle:'dotted',
                 textUnderlineOffset:'2px', color:col, fontWeight:600 }}
      >{label}</span>
    );
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes.length ? nodes : [text];
}

function buildReasoningFromBreakdown(score: number, grade: string, breakdown: Record<string,number>): string {
  // Intentional low-light / chiaroscuro / soft-focus detection.
  // Strong Narrative intent alongside lower Lighting CLIP score = deliberate mood.
  const narrativeScore = (breakdown['Narrative'] ?? 0) as number;
  const lightingScore  = (breakdown['Lighting']  ?? 1.0) as number;
  const isMoody = narrativeScore >= 0.38 && lightingScore < 0.55;

  const NOTES_LIGHTING: [number,string][] = isMoody
    ? [[0.78,"Light has direction and authority — shadow play is doing the work."],[0.62,"Light is readable; contrast holds."],[0.45,"Low-key rendering — shadow weight reads as intentional mood."],[0,"Deep shadow dominance. Chiaroscuro or available-light approach — darkness as intent."]]
    : [[0.78,"Light has direction and authority — shadow play is doing the work."],[0.62,"Light is readable; contrast holds."],[0.45,"Flat light. No drama, no depth — nothing to push the subject forward."],[0,"Light is fighting the image. Blown highlights or dead-flat exposure."]];

  const NOTES_TECHNICAL: [number,string][] = isMoody
    ? [[0.78,"Technical execution disappears into the image — as it should."],[0.62,"Technically clean. No distraction."],[0.45,"Soft rendering or organic grain — intentional aesthetic signature, not a failure."],[0,"Technical compromise is visible — in fine-art work, intentional grain and glow are valid."]]
    : [[0.78,"Technical execution disappears into the image — as it should."],[0.62,"Technically clean. No distraction."],[0.45,"Some softness or exposure drift. Manageable, not invisible."],[0,"Technical failure is visible — motion blur, clipping, or heavy noise."]];

  const NOTES: Record<string, [number,string][]> = {
    Composition:     [[0.78,"Frame is airtight — every element earns its place."],[0.62,"Geometry works; the eye moves without fighting the edges."],[0.45,"Framing is serviceable but the edges carry dead weight."],[0,"Frame is loose — crop it or reshoot it."]],
    Lighting:        NOTES_LIGHTING,
    Narrative:       [[0.78,"The moment is decisive — gesture or tension frozen at exactly the right frame."],[0.62,"A moment caught, not staged — feels authentic."],[0.45,"Something is happening but nothing is at stake."],[0,"No moment. The scene is static and the camera just witnessed it."]],
    'Human/Culture': [[0.78,"The human subject commands the frame — presence is undeniable."],[0.62,"Human element adds weight; the figure belongs here."],[0.45,"Figures are present but incidental — they don't anchor anything."],[0,"No human element. Architectural or environmental — works only if intentional."]],
    Technical:       NOTES_TECHNICAL,
  };

  const tier = grade?.includes('Strong') ? 'strong' : grade?.includes('Weak') ? 'weak' : 'mid';
  // Use the actual graded score for display — this is the final fused pipeline value.
  // avgScore is kept for internal note selection only (moody/chiaroscuro detection).
  const bdVals = Object.values(breakdown).filter(v => typeof v === 'number') as number[];
  const avgScore = bdVals.length > 0 ? bdVals.reduce((s, v) => s + v, 0) / bdVals.length : score;
  const pct  = Math.round(score * 100);
  // Only rank real photographic aspects (keys with copy in NOTES). Excludes private
  // metadata and technical-audit fields so the Best/Weakest footer never names jargon.
  const sorted = Object.entries(breakdown)
    .filter(([k,v]) => typeof v === 'number' && k in NOTES)
    .sort((a,b) => b[1]-a[1]);
  const topKey    = sorted[0]?.[0]       ?? 'Narrative';
  const bottomKey = sorted.at(-1)?.[0]   ?? 'Technical';

  const VERDICT_LIGHTING_WEAK = isMoody
    ? "Atmospheric depth through shadow — low-key is the visual language here."
    : "Light is the problem here, not the solution.";
  const VERDICT_LIGHTING_MID = isMoody
    ? "Shadow and atmosphere doing most of the work — mood over exposure."
    : "Light is present but not working hard enough.";

  const VERDICT: Record<string, Record<string,string>> = {
    strong: { Narrative:"Street photographer's instinct — right place, right frame, right moment.", Composition:"Geometric authority. The structure carries the image.", Lighting:"Light as subject. Everything else serves the atmosphere.", 'Human/Culture':"The figure is the photograph. Everything else is context.", Technical:"Technically confident — the craft is invisible." },
    mid:    { Narrative:"The moment is there but the frame doesn't fully commit to it.", Composition:"Decent bones. The structure works but doesn't surprise.", Lighting: VERDICT_LIGHTING_MID, 'Human/Culture':"The human element is in the frame but not in control of it.", Technical:"Technically adequate. Won't lose the shot but won't win it either." },
    weak:   { Narrative:"No decisive moment — the shutter fired but nothing was caught.", Composition:"The frame is not resolved. Too much, too little, or in the wrong place.", Lighting: VERDICT_LIGHTING_WEAK, 'Human/Culture':"The subject is lost. Distance, angle, or timing killed it.", Technical:"Technical compromise dominates. The image can't recover from it." },
  };

  const LBL: Record<string,string> = { Narrative:'Moment', 'Human/Culture':'Human' };
  const verdict = VERDICT[tier]?.[topKey] ?? '';
  const lines: string[] = [tier[0].toUpperCase()+tier.slice(1)];
  if (verdict) lines.push(verdict);
  lines.push('');
  for (const [k,v] of sorted) {
    const note = (NOTES[k] ?? []).find(([t]) => v >= t)?.[1] ?? '';
    if (note) lines.push(`${LBL[k]??k}: ${note}`);
  }
  lines.push(`\nBest: ${LBL[topKey]??topKey}   ·   Weakest: ${LBL[bottomKey]??bottomKey}`);
  return lines.join('\n');
}

/* Aspect -> canonical dimension classifier moved to lib/aspects.ts - shared by App and the analysis panel. */

/* ── Factor Annotations overlay ────────────────────────────────── */
const REGION_BOX: Record<string, [number,number,number,number]> = {
  'top-third':    [0,      0,      1,    0.33],
  'center':       [0.2,    0.2,    0.6,  0.6],
  'bottom-third': [0,      0.67,   1,    0.33],
  'full':         [0,      0,      1,    1],
  'left-half':    [0,      0,      0.5,  1],
  'right-half':   [0.5,    0,      0.5,  1],
  'top-left':     [0,      0,      0.5,  0.5],
  'top-right':    [0.5,    0,      0.5,  0.5],
  'bottom-left':  [0,      0.5,    0.5,  0.5],
  'bottom-right': [0.5,    0.5,    0.5,  0.5],
};
/* Annotation marks are the MACHINE's, not the photographer's, so they stay cold
 * or neutral — --mark is reserved for his own rings and stars, and a factor
 * overlay wearing it would claim his authorship. */
const FACTOR_COLORS: Record<string, string> = {
  blur:    T.alarmWarn,
  heatmap: T.alarmCrit,
  grid:    T.ink2,
};

const _INK  = T.ink;
const _SH   = `0 0 8px ${T.well}, 0 1px 3px ${T.well}`;
const _MONO = "var(--font-mono)";

function FactorAnnotations({ factors }: { factors: any[] }) {
  if (!factors || factors.length === 0) return null;
  return (
    <svg
      style={{ position:'absolute', top:0, left:0, width:'100%', height:'100%', pointerEvents:'none' }}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <filter id="fa-txt" x="-5%" y="-20%" width="110%" height="140%">
          <feDropShadow dx="0" dy="0" stdDeviation="2.5" floodColor={T.well} floodOpacity="1"/>
        </filter>
      </defs>
      {factors.map((f: any, i: number) => {
        const [bx, by, bw, bh] = REGION_BOX[f.region] ?? REGION_BOX['full'];
        const isStrength = (f.impact ?? 0) > 0;
        const isWeakness = (f.impact ?? 0) < 0;
        const color = isStrength ? T.gradeStrong : isWeakness ? T.alarmCrit : (FACTOR_COLORS[f.type] ?? T.ink2);

        // Region in %, clamped so edges are always inside the frame
        const rx = bx * 100, ry = by * 100, rw = bw * 100, rh = bh * 100;
        const rx2 = rx + rw, ry2 = ry + rh;
        // Center point
        const cx = rx + rw / 2, cy = ry + rh / 2;

        // Label: inside top-left of region, always on-screen
        const lx = Math.max(1, Math.min(rx + 1.5, 70));
        const ly = Math.max(4, ry + 5);

        const impactAbs = Math.round(Math.abs(f.impact ?? 0) * 100);
        const badge = isStrength ? `[+${impactAbs}]` : isWeakness ? `[-${impactAbs}]` : null;
        const glyph = isStrength ? '●' : isWeakness ? '✕' : '○';

        return (
          <g key={i}>
            {/* Region fill wash */}
            <rect x={`${rx}%`} y={`${ry}%`} width={`${rw}%`} height={`${rh}%`}
              fill={color} fillOpacity="0.10" stroke="none"/>

            {/* Region border */}
            <rect x={`${rx}%`} y={`${ry}%`} width={`${rw}%`} height={`${rh}%`}
              fill="none" stroke={color} strokeWidth="1.8" strokeOpacity="0.75"
              strokeDasharray="6 3"/>

            {/* Strength: bold dashed ellipse around region */}
            {isStrength && (
              <ellipse cx={`${cx}%`} cy={`${cy}%`}
                rx={`${rw * 0.44}%`} ry={`${rh * 0.42}%`}
                fill="none" stroke={T.gradeStrong} strokeWidth="2.2"
                strokeOpacity="0.85" strokeDasharray="7 4"/>
            )}

            {/* Weakness: bold X-strike */}
            {isWeakness && (<>
              <line x1={`${rx + rw*0.06}%`} y1={`${ry + rh*0.06}%`}
                    x2={`${rx + rw*0.94}%`} y2={`${ry + rh*0.94}%`}
                stroke={T.alarmCrit} strokeWidth="2.2" strokeOpacity="0.8"/>
              <line x1={`${rx + rw*0.94}%`} y1={`${ry + rh*0.06}%`}
                    x2={`${rx + rw*0.06}%`} y2={`${ry + rh*0.94}%`}
                stroke={T.alarmCrit} strokeWidth="2.2" strokeOpacity="0.8"/>
            </>)}

            {/* Center dot */}
            <circle cx={`${cx}%`} cy={`${cy}%`} r="0.6%"
              fill={color} fillOpacity="1"/>

            {/* Connector: center dot → label */}
            <line x1={`${cx}%`} y1={`${cy}%`} x2={`${lx}%`} y2={`${ly}%`}
              stroke={color} strokeWidth="1" strokeOpacity="0.5" strokeDasharray="3 3"/>

            {/* Label: glyph + name + badge, inside region */}
            <text x={`${lx}%`} y={`${ly}%`}
              fill={color} fontSize="12" fontWeight="700"
              fontFamily="var(--font-mono)"
              filter="url(#fa-txt)">
              {glyph} {f.label}{badge ? ` ${badge}` : ''}
            </text>

            {/* Note on second line */}
            {f.note && (
              <text x={`${lx}%`} y={`${Math.min(ly + 4, ry2 - 2)}%`}
                fill={color} fontSize="10" fillOpacity="0.75"
                fontFamily="var(--font-mono)"
                filter="url(#fa-txt)">
                {f.note}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/* ── Analysis HUD — pen-notation corner annotation on the image ──── */
function AnalysisHUD({ grade, score, breakdown }: { grade: string; score: number; breakdown: Record<string,number> }) {
  const ASPECT_KEYS = ['Technical','Composition','Lighting','Narrative','Human/Culture'];
  const aspects = ASPECT_KEYS.map(k => [k, (breakdown[k] ?? 0)] as [string,number]);
  const maxV = Math.max(...aspects.map(([,v]) => v));
  const minV = Math.min(...aspects.map(([,v]) => v));
  const pct  = Math.round(score * 100);
  const gradeColor = gc(grade);
  return (
    <div style={{ position:'absolute', bottom:62, left:20, pointerEvents:'none', zIndex:2 }}>
      {/* Score + grade written directly on the photo */}
      <div style={{ display:'flex', alignItems:'baseline', gap:6, marginBottom:10 }}>
        <span style={{ fontFamily:_MONO, fontSize:'var(--text-xl)', fontWeight:500, lineHeight:'var(--leading-none)',
          color:gradeColor, textShadow:_SH, letterSpacing:'var(--track-tight)' }}>{pct}</span>
        <span style={{ fontFamily:_MONO, fontSize:'var(--text-xs)', color:_INK, textShadow:_SH, opacity:.45 }}>/100</span>
        <span style={{ fontFamily:_MONO, fontSize:'var(--text-xs)', fontWeight:500, letterSpacing:'var(--track-label)',
          color:gradeColor, textShadow:_SH, marginLeft:4 }}>{gradeLabel(grade).toUpperCase()}</span>
      </div>
      {/* Aspect scores as pen-ruled tick lines */}
      <div style={{ display:'flex', flexDirection:'column', gap:5 }}>
        {aspects.map(([k, v]) => {
          const vpct = Math.round((v as number) * 100);
          const isTop = v === maxV;
          const isBot = v === minV && v !== maxV;
          const col = isTop ? T.gradeStrong : isBot ? T.gradeWeak : _INK;
          const filled = vpct * 0.76;
          return (
            <div key={k} style={{ display:'flex', alignItems:'center', gap:8 }}>
              <span style={{ fontFamily:_MONO, fontSize:'var(--text-xs)', color:col, textShadow:_SH,
                width:82, textAlign:'right', flexShrink:0, opacity:.85, letterSpacing:'var(--track-body)' }}>
                {k}
              </span>
              <svg width="80" height="9" style={{ flexShrink:0, overflow:'visible' }}>
                <line x1="0" y1="4.5" x2="76" y2="4.5" stroke={`${col}`} strokeWidth="0.75" opacity="0.25"/>
                <line x1="0" y1="4.5" x2={filled} y2="4.5" stroke={col} strokeWidth="1.5" strokeLinecap="round"/>
                <line x1={filled} y1="1" x2={filled} y2="8" stroke={col} strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              <span style={{ fontFamily:_MONO, fontSize:'var(--text-xs)', fontWeight:500, color:col,
                textShadow:_SH, flexShrink:0, width:22 }}>{vpct}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Niche registry (mirrors src/niche_registry.py) ─────────────── */
const NICHE_GROUPS = [
  { category: "Street & Documentary", niches: [
    { key: "classic_street",  label: "Classic Street" },
    { key: "documentary",     label: "Documentary" },
    { key: "photojournalism", label: "Photojournalism" },
    { key: "travel_cultural", label: "Travel & Cultural" },
  ]},
  { category: "Architecture & Space", niches: [
    { key: "architectural", label: "Architectural" },
    { key: "liminal",       label: "Liminal / Atmospheric" },
    { key: "urban_city",    label: "Urban & City" },
  ]},
  { category: "Light & Mood", niches: [
    { key: "night",         label: "Night Photography" },
    { key: "long_exposure", label: "Long Exposure" },
    { key: "fine_art",      label: "Fine Art" },
    { key: "minimalist",    label: "Minimalist" },
  ]},
  { category: "Subject-Focused", niches: [
    { key: "portrait",          label: "Portrait" },
    { key: "wildlife",          label: "Wildlife" },
    { key: "fashion_editorial", label: "Fashion & Editorial" },
    { key: "sports_action",     label: "Sports & Action" },
    { key: "wedding",           label: "Wedding" },
  ]},
  { category: "Creative & Specialized", niches: [
    { key: "landscape",    label: "Landscape" },
    { key: "abstract",     label: "Abstract" },
    { key: "macro",        label: "Macro / Close-up" },
    { key: "experimental", label: "Experimental" },
  ]},
];

/* ── App ────────────────────────────────────────────────────────── */
export default function App() {
  const [folder,     setFolder]     = useState("");
  const [preset,     setPreset]     = useState("classic_street");
  const [photos,     setPhotos]     = useState<any[]>([]);
  const [carousel,   setCarousel]   = useState<any[]>([]);
  const [saved,      setSaved]      = useState<{name: string; sequence: any[]}[]>([]);
  const [loading,      setLoading]      = useState(false);
  const [listLoading,  setListLoading]  = useState(false);
  const [gradeProgress, setGradeProgress] = useState(0);
  const [gradeDesc,     setGradeDesc]     = useState("");
  // Encoder quality tier chosen for this run ("Fast" | "Balanced" | "Pro").
  // Deliberately a plain quality word — never a model name.
  const [gradeQuality,  setGradeQuality]  = useState("");
  const [gradeStartMs,  setGradeStartMs]  = useState<number | null>(null);
  const [gradeEtaSecs,  setGradeEtaSecs]  = useState<number | null>(null);
  const [toast,      setToast]      = useState<{msg: string; type: "success"|"error"|"info"} | null>(null);
  const [catalogSaveFailed, setCatalogSaveFailed] = useState(false);
  const [selId,      setSelId]      = useState<string | null>(null);
  const [nicheRec,   setNicheRec]   = useState<any>(null);
  const [nicheDetecting, setNicheDetecting] = useState(false);
  const [infoTab,    setInfoTab]    = useState<"exif"|"breakdown"|"analysis">("breakdown");
  const [scanMode,   setScanMode]   = useState(false);
  const [deepGrade,  setDeepGrade]  = useState(false);   // OFF = fast SigLIP zero-shot; ON = Qwen VLM (slower, GPU)
  const [graderUsed, setGraderUsed] = useState<'fast'|'deep'|'scan'|null>(null);  // which grader actually ran (transparency badge)
  const [mainTab,    setMainTab]    = useState<"gallery"|"duplicates"|"creative">("gallery");
  /* Duplicates view: groups render incrementally (4 at a time via a scroll
   * sentinel) so 119 groups × ~150 thumbs never mount at once. */
  const [dupGroupsShown, setDupGroupsShown] = useState(4);
  useEffect(() => { setDupGroupsShown(4); }, [mainTab]);
  const revealMoreDupGroups = useCallback(() => setDupGroupsShown(n => n + 4), []);
  /* RAM chip popover — the crit chip must offer an action, not just an alarm. */
  const [ramPopoverOpen, setRamPopoverOpen] = useState(false);
  const [seqMode,    setSeqMode]    = useState<'auto'|'director'|'story'|'competition'>('story');
  const [directorPrompt,  setDirectorPrompt]  = useState('');
  const [directorResult,  setDirectorResult]  = useState<any>(null);
  const [directorLoading, setDirectorLoading] = useState(false);
  const [directorPool,    setDirectorPool]    = useState<any[]>([]);
  const [mogcoTarget]     = useState(5);   // default story length; story building lives in the Story tab, not culling
  const [mogcoMinScore,   setMogcoMinScore]   = useState(0.45);
  const [uploadLoading,   setUploadLoading]   = useState(false);
  const [uploadDragOver,  setUploadDragOver]  = useState(false);
  const [loupeMode,  setLoupeMode]  = useState<"loupe"|"grid">("loupe");
  const [subjType,   setSubjType]   = useState<string | null>(null);
  const [locked,     setLocked]     = useState<Set<string>>(new Set());
  const [used,       setUsed]       = useState<Set<string>>(new Set());
  const [redacted,      setRedacted]      = useState<Set<string>>(new Set());
  const [showDuplicates,setShowDuplicates] = useState(false);
  const [folders,      setFolders]      = useState<string[]>([]);
  const [browserMode,  setBrowserMode]  = useState<'open'|'add'>('open');
  const [catalogBanner,setCatalogBanner]= useState(false);
  const saveTimerRef       = useRef<ReturnType<typeof setTimeout> | null>(null);
  const skipFolderLoadRef  = useRef(false);
  const [showBrowser,setShowBrowser]= useState(false);
  const [bPath,      setBPath]      = useState("C:\\Users");
  const [bFolders,   setBFolders]   = useState<string[]>([]);
  const [bImages,    setBImages]    = useState<string[]>([]);
  const [bSelFolders, setBSelFolders] = useState<Set<string>>(new Set());
  const [lastBClick, setLastBClick] = useState<number | null>(null);
  const [bLoading,   setBLoading]   = useState(false);
  const [copied,     setCopied]     = useState(false);
  const [rightW,     setRightW]     = useState(280);
  const [filmThumbH, setFilmThumbH] = useState(84);
  const [showFilename,setShowFilename] = useState(true);
  const [showTweaks, setShowTweaks] = useState(false);
  const [filterGrade,setFilterGrade] = useState<string | null>(null);
  const [filterStars,setFilterStars] = useState<number | null>(null);
  const [sortScore,  setSortScore]   = useState<'desc'|'asc'|null>(null);
  const [exportModal,setExportModal] = useState(false);
  const [selectedIds,setSelectedIds] = useState<Set<string>>(new Set());
  const [selectMode, setSelectMode]  = useState(false);
  const [showStarSort, setShowStarSort] = useState(false);
  const [seqMinStars, setSeqMinStars]   = useState(0);
  const [dragOver,    setDragOver]      = useState(false);
  const [backendReady,   setBackendReady]   = useState(false);
  const [backendError,   setBackendError]   = useState(false);
  const [graderStatus,   setGraderStatus]   = useState<{last_mode:string,draft_available:boolean,verify_available:boolean,last_error:string|null,qwen_warm:boolean,qwen_loading:boolean,qwen_download_pct:number|null,warmup_done:boolean,warmup_running:boolean,compute_device?:string,vram_free_gb?:number|null,vram_total_gb?:number|null,gpu_name?:string|null,ram_free_gb?:number|null,ram_total_gb?:number|null,ram_min_gb?:number}|null>(null);
  // Live system-memory snapshot, polled every 2 s (see /api/system/ram) so the RAM
  // readiness indicator tracks Task Manager in real time rather than refreshing
  // only on modal open.
  const [sysRam, setSysRam] = useState<{ram_free_gb:number|null,ram_total_gb:number|null,ram_percent:number|null,ram_min_gb:number}|null>(null);
  const [preGradeModal,  setPreGradeModal]  = useState<{photoCount:number}|null>(null);
  const preGradeDialogRef = useRef<HTMLDivElement>(null);
  // Modal accessibility: Escape closes, Tab is trapped inside the dialog, and
  // focus is restored to the trigger on close (WCAG 2.1.2 / 2.4.3).
  useEffect(() => {
    if (!preGradeModal) return;
    const prevFocus = document.activeElement as HTMLElement | null;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { setPreGradeModal(null); return; }
      if (e.key !== 'Tab') return;
      const root = preGradeDialogRef.current;
      if (!root) return;
      const nodes = root.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );
      if (!nodes.length) return;
      const first = nodes[0], last = nodes[nodes.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('keydown', onKey); prevFocus?.focus?.(); };
  }, [preGradeModal]);
  const [rescanAll,      setRescanAll]      = useState(true);
  const [heatmapB64,     setHeatmapB64]     = useState<string | null>(null);
  const [heatmapPath,    setHeatmapPath]    = useState<string | null>(null);
  const [showHeatmap,    setShowHeatmap]    = useState(false);
  const [critTrigger,    setCritTrigger]    = useState<string>('');
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [pegFile,        setPegFile]        = useState<File | null>(null);
  const [pegHash,        setPegHash]        = useState<string | null>(null);
  const [pegLoading,     setPegLoading]     = useState(false);
  // ── Semantic search state ─────────────────────────────────────────────────
  const [searchQuery,    setSearchQuery]    = useState("");
  const [searchResults,  setSearchResults]  = useState<Set<string> | null>(null); // Set of paths
  const [searchLoading,  setSearchLoading]  = useState(false);
  // ── Jury critique state ───────────────────────────────────────────────────
  const [juryLoading,    setJuryLoading]    = useState(false);
  const [juryCritique,   setJuryCritique]   = useState<string | null>(null);
  const [juryThink,      setJuryThink]      = useState<string | null>(null);
  const [juryCritPath,   setJuryCritPath]   = useState<string | null>(null);
  // ── Engine health state ───────────────────────────────────────────────────
  const [engineHealth,        setEngineHealth]        = useState<{ status: "checking"|"online"|"offline"; missing: string[] }>({ status: "checking", missing: [] });
  const [ollamaPs,            setOllamaPs]            = useState<{name:string; size_vram:number; size_total:number}[]>([]);
  const [bannerDismissed,     setBannerDismissed]     = useState(false);
  const [isDownloading,       setIsDownloading]       = useState(false);
  const [downloadProgress,    setDownloadProgress]    = useState(0);
  const [currentDownloadModel,setCurrentDownloadModel]= useState("");
  const [downloadError,       setDownloadError]       = useState<string | null>(null);
  const [updateRequired,      setUpdateRequired]      = useState(false);
  // ── Creative Direction state ──────────────────────────────────────────────
  const [creativeAnchor,   setCreativeAnchor]   = useState<string | null>(null);
  const [creativePrompt,   setCreativePrompt]   = useState("");
  const [creativeMode,     setCreativeMode]     = useState<"canny"|"depth">("canny");
  // 4-10. Below 4 there is no sequence to speak of; above 10 the set stops
  // holding together. Default 7. Maps straight onto the API's n_target.
  const [creativeCount,    setCreativeCount]    = useState(7);
  const [creativeLoading,  setCreativeLoading]  = useState(false);
  const [creativeProgress, setCreativeProgress] = useState(0);
  const [creativeStage,    setCreativeStage]    = useState("");
  const [creativeResults,     setCreativeResults]     = useState<any[]>([]);
  const [creativeOutDir,      setCreativeOutDir]      = useState("");
  // Non-empty when the sequence was NOT art-directed: a score sort wearing a
  // story’s clothes. Shown, never swallowed.
  const [creativeFallback,    setCreativeFallback]    = useState("");
  // How tightly the chosen set hangs together, reported by story_selector.
  // Deliberately a readout, not a gate: no cohesion floor could be justified
  // without grading on a curve, so the number is shown and the user judges.
  const [creativeSelection,   setCreativeSelection]   = useState<any>(null);
  const [creativeShowOriginal,setCreativeShowOriginal]= useState(false);
  const [usedCount,           setUsedCount]           = useState(0);
  const [sequenceSaving,      setSequenceSaving]      = useState(false);
  // ── PDF RAG state ─────────────────────────────────────────────────────────
  const [ragPdfs,       setRagPdfs]       = useState<{name:string,pages:number,phrases:number}[]>([]);
  const [ragUploading,  setRagUploading]  = useState(false);
  // ── Auditor / XAI overlay state ───────────────────────────────────────────
  const [isAuditModeActive,     setIsAuditModeActive]     = useState(false);
  const [reasoningOverlayUrl,   setReasoningOverlayUrl]   = useState<string | null>(null);
  const [reasoningOverlayPath,  setReasoningOverlayPath]  = useState<string | null>(null);
  const [showEyeOverlay,        setShowEyeOverlay]        = useState(false);
  const [photoNatDims,          setPhotoNatDims]          = useState<{w:number;h:number}|null>(null);
  const [deepCritique,          setDeepCritique]          = useState<{narrative_arc:string;geometry_composition:string}|null>(null);
  const [deepCritiqueLoading,   setDeepCritiqueLoading]   = useState(false);
  /* Loupe full-preview failure. /api/photo falls back to a RAW full decode;
   * when even that fails the stage used to go silently black. Now the loupe
   * degrades to the thumbnail with an honest note and a real retry. */
  const [loupePreviewFailed,    setLoupePreviewFailed]    = useState(false);
  const [loupeRetry,            setLoupeRetry]            = useState(0);
  useEffect(() => { setLoupePreviewFailed(false); setLoupeRetry(0); }, [selId]);

  /* filmRef removed — the filmstrip is extracted to <Filmstrip>, which owns
   * its own ref and index-based auto-scroll. */
  const dragCounter = useRef(0);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  const notify = useCallback((msg: string, type: "success"|"error"|"info" = "info") =>
    setToast({ msg, type }), []);

  useEffect(() => {
    if (!toast) return;
    // Errors need longer — they usually name something the user must act on,
    // and 3.2 s is not enough to read a path or a reason.
    const t = setTimeout(() => setToast(null), toast.type === "error" ? 10_000 : 3200);
    return () => clearTimeout(t);
  }, [toast]);

  /* live ETA countdown while grading is in progress */
  const gradeProgressRef = useRef(gradeProgress);
  gradeProgressRef.current = gradeProgress;
  useEffect(() => {
    if (!loading || gradeStartMs === null) {
      setGradeEtaSecs(null);
      return;
    }
    const id = setInterval(() => {
      const p = gradeProgressRef.current;
      if (p <= 0.02) return;
      const elapsed = (Date.now() - gradeStartMs) / 1000;
      const total   = elapsed / p;
      setGradeEtaSecs(Math.max(0, Math.round(total - elapsed)));
    }, 1000);
    return () => clearInterval(id);
  }, [loading, gradeStartMs]);

  /* poll backend until it responds — shows loading screen until ready */
  useEffect(() => {
    let cancelled = false;
    let timerId: ReturnType<typeof setTimeout>;
    let attempts = 0;
    const check = () => {
      attempts++;
      if (attempts > 100) { // 60 s timeout
        if (!cancelled) setBackendError(true);
        return;
      }
      fetch(`${API}/`)
        .then(r => {
          // Retry on ANY non-ready outcome — a non-OK response used to stop the
          // loop silently and leave the app on "Starting…" forever.
          if (r.ok && !cancelled) setBackendReady(true);
          else if (!cancelled && attempts <= 100) timerId = setTimeout(check, 600);
        })
        .catch(() => {
          if (!cancelled) timerId = setTimeout(check, 600);
        });
    };
    check();
    return () => { cancelled = true; clearTimeout(timerId); };
  }, []);

  /* poll Ollama engine health every 10 s */
  const fetchEngineHealth = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/health/engine`);
      if (r.ok) {
        const d = await r.json();
        const status = d.status ?? "offline";
        if (status === "offline") setBannerDismissed(false);
        setEngineHealth({ status, missing: d.missing_models ?? [] });
      } else {
        setBannerDismissed(false);
        setEngineHealth({ status: "offline", missing: [] });
      }
    } catch {
      setBannerDismissed(false);
      setEngineHealth({ status: "offline", missing: [] });
    }
  }, []);

  useGuardedInterval(fetchEngineHealth, 10_000, [fetchEngineHealth]);

  const fetchOllamaPs = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/ollama/status`);
      if (r.ok) {
        const d = await r.json();
        setOllamaPs(d.models ?? []);
      } else {
        setOllamaPs([]);
      }
    } catch {
      setOllamaPs([]);
    }
  }, []);

  useGuardedInterval(fetchOllamaPs, 15_000, [fetchOllamaPs]);

  // Live RAM poll — cheap psutil-only endpoint, every 2 s, paused while hidden.
  const fetchSysRam = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/system/ram`);
      if (r.ok) setSysRam(await r.json());
    } catch { /* leave last reading */ }
  }, []);
  useGuardedInterval(fetchSysRam, 2_000, [fetchSysRam]);

  const fetchRagStatus = useCallback(async () => {
    try {
      const resp = await axios.get<{phrases: string[], pdfs: {name:string,pages:number,phrases:number}[]}>(`${API}/api/rag/concepts`);
      setRagPdfs(resp.data.pdfs ?? []);
    } catch { /* silent */ }
  }, []);

  useEffect(() => { fetchRagStatus(); }, [fetchRagStatus]);

  /* Fetch the optional GGUF models the app is missing.

     This used to filter engineHealth.missing for names NOT ending in .gguf,
     because in the Ollama era those were Ollama tags. Ollama was removed; the
     filter survived. The registry offers nothing BUT .gguf, so the filter
     emptied the list every time and the button silently did nothing -- leaving
     Story mode permanently on its score-sort fallback on any fresh install.

     The server takes the keyword "optional" and runs
     scripts/fetch_models.py --with-optional --json, which streams ndjson
     records of {name, status, message}. It is not Ollama's pull protocol, so
     the old total/completed/status:"success" parsing never matched either. */
  const handleDownloadMissing = useCallback(async () => {
    const missing = engineHealth.missing ?? [];
    if (missing.length === 0) return;
    setIsDownloading(true);
    setDownloadError(null);
    setUpdateRequired(false);
    let failure: string | null = null;
    let done = 0;
    try {
      setCurrentDownloadModel('optional models');
      setDownloadProgress(0);
      const resp = await fetch(`${API}/api/models/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: 'optional' }),
      });
      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { done: finished, value } = await reader.read();
        if (finished) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.trim()) continue;
          let rec: any;
          try { rec = JSON.parse(line); } catch { continue; }
          if (rec.name) setCurrentDownloadModel(String(rec.name));
          if (rec.status === 'fail') failure = String(rec.message ?? rec.name);
          if (rec.status === 'ok' || rec.status === 'skip') {
            done += 1;
            setDownloadProgress(Math.min(99, Math.round((done / Math.max(missing.length, 1)) * 100)));
          }
        }
      }
      setDownloadProgress(100);
    } catch (e: any) {
      failure = e?.message ?? 'Could not reach the model downloader.';
    }
    setIsDownloading(false);
    setCurrentDownloadModel('');
    if (failure) {
      setDownloadError(failure);
      notify(`Could not download the models. ${failure}`, 'error');
    } else {
      notify('Models installed', 'success');
      fetchEngineHealth();
    }
  }, [engineHealth.missing, fetchEngineHealth, notify]);

  /* fetch grader model status on startup and after each grading run */
  const isDoneForStatus = !loading && photos.length > 0 && photos.some((p:any) => p.grade !== 'Pending');
  useEffect(() => {
    if (!backendReady) return;
    fetch(`${API}/api/models/status`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setGraderStatus(d); })
      .catch(() => {});
  }, [backendReady, isDoneForStatus, preGradeModal]);

  /* Poll status every 3 s while the pre-grade modal is open — keeps the RAM /
     engine readiness live so the user sees an up-to-date "clear to grade" state. */
  useEffect(() => {
    if (!preGradeModal) return;
    const id = setInterval(() => {
      fetch(`${API}/api/models/status`)
        .then(r => r.ok ? r.json() : null)
        .then(d => { if (d) setGraderStatus(d); })
        .catch(() => {});
    }, 3000);
    return () => clearInterval(id);
  }, [preGradeModal]);

  /* fetch excluded-photo count from the server */
  useEffect(() => {
    fetch(`${API}/api/creative-direction/used-count`)
      .then(r => r.json())
      .then(d => setUsedCount(d.count ?? 0))
      .catch(() => {});
  }, []);

  const sel = useMemo(() => photos.find(p => p.id === selId) ?? photos[0] ?? null, [photos, selId]);

  /* Duplicate group stats — the single source for both the Duplicates tab
   * count and the Similar Shots header, so the two numbers always agree.
   * (They used to be computed two different ways and disagreed by ~340.) */
  const dupStats = useMemo(() => {
    const byCluster: Record<number, any[]> = {};
    for (const p of photos) {
      if (p.cluster_id < 0) continue;
      (byCluster[p.cluster_id] ??= []).push(p);
    }
    const groups = Object.values(byCluster)
      .map(g => {
        const best = g.find(p => (p.sim_flag||'').includes('Best')) ?? g[0];
        const rest = g.filter(p => p !== best).sort((a,b) => b.score - a.score);
        return { best, rest, all: [best, ...rest] };
      })
      .sort((a, b) => b.all.length - a.all.length);
    return { groups, totalDups: groups.reduce((s, g) => s + g.rest.length, 0) };
  }, [photos]);

  useEffect(() => {
    if (photos.length > 0 && !selId) setSelId(photos[0].id);
  }, [photos]);

  /* Lazy EXIF fetch.
   *
   * "Has no EXIF yet" was the wrong test. A catalogue written before the reader
   * was rewritten holds the old sparse shape — often just date and time — and
   * that is non-empty, so the panel would show two rows forever and never ask
   * the server for the other twenty. `file_size` is the marker: it comes from
   * stat() rather than the file's EXIF block, so the current reader emits it for
   * every readable photo and no older record can have it. */
  useEffect(() => {
    if (!sel) return;
    const cached = sel.exif || {};
    if (Object.keys(cached).length > 0 && cached.file_size) return;
    const ctrl = new AbortController();
    axios.get(`${API}/api/exif`, { params: { path: sel.path }, signal: ctrl.signal })
      .then(r => {
        if (Object.keys(r.data).length > 0)
          setPhotos(prev => prev.map(p => p.id === sel.id ? { ...p, exif: r.data } : p));
      })
      .catch(e => { if (!axios.isCancel(e)) console.warn('[exif]', e); });
    return () => ctrl.abort();
  }, [sel?.id]);

  /* auto-scroll filmstrip to selected thumb — moved into <Filmstrip> (index-based,
   * works when the selection sits outside the rendered window) */


  const filteredPhotos = useMemo(() => {
    const carouselPaths = new Set(carousel.map((c: any) => c.path));
    const base = photos.filter(p => {
      if (searchResults !== null && !searchResults.has(p.path)) return false; // semantic search filter
      if (!showDuplicates && redacted.has(p.path)) return false;   // non-best duplicates hidden unless toggled
      const starsOk = filterStars === null || p.stars === filterStars;
      if (filterGrade) return gradeLabel(p.grade) === filterGrade && starsOk;
      if (carouselPaths.has(p.path)) return true;                   // sequence photos always visible when no grade filter
      return starsOk;
    });
    if (!sortScore) return base;
    return [...base].sort((a, b) => sortScore === 'desc' ? b.score - a.score : a.score - b.score);
  }, [photos, filterGrade, filterStars, redacted, showDuplicates, sortScore, carousel, searchResults]);

  /* Star rating — declared BEFORE the keyboard effect below: that effect lists
   * handleSetStars in its dependency array, and deps arrays evaluate during
   * render. A const declared later would throw "Cannot access 'handleSetStars'
   * before initialization" (TDZ) the moment this view mounts. */
  const handleSetStars = useCallback((id: string, stars: number) => {
    setPhotos(prev => prev.map(p => p.id === id ? { ...p, stars } : p));
    // Fire-and-forget: train PersonalHead + queue DPO event
    const path = photos.find(p => p.id === id)?.path;
    if (path) {
      axios.post(`${API}/api/personal/star`, { path, stars }).catch(() => {});
    }
  }, [photos]);

  /* keyboard nav — full culling flow, no mouse required.
   *
   *   ←/→ or h/l   move selection          g / e   grid ⇄ loupe
   *   1–5          star (repeat = clear)   0       clear stars
   *   x            toggle "used" (persisted to photo_flags.json)
   *
   * The used-toggle posts to /api/flags/used so the mark survives restarts,
   * matching how the flags are loaded at boot. Deps include handleSetStars and
   * sel: both were previously captured stale by this effect's closure. */
  const handleToggleUsedKb = useCallback((id: string) => {
    const path = photos.find(p => p.id === id)?.path;
    if (!path) return;
    let nextUsed: Set<string> | null = null;
    setUsed(prev => {
      nextUsed = new Set(prev);
      nextUsed.has(path) ? nextUsed.delete(path) : nextUsed.add(path);
      return nextUsed;
    });
    // Persist after the state update resolves; fire-and-forget like the other flags.
    setTimeout(() => {
      axios.post(`${API}/api/flags/used`, { path }).catch(() => {});
    }, 0);
  }, [photos]);

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (['INPUT','SELECT','TEXTAREA'].includes((document.activeElement as HTMLElement)?.tagName)) return;
      const ids = filteredPhotos.map(p => p.id);
      const i = ids.indexOf(selId ?? '');
      if (e.key === 'ArrowRight' || e.key === 'l') { e.preventDefault(); if (i < ids.length-1) setSelId(ids[i+1]); }
      if (e.key === 'ArrowLeft'  || e.key === 'h') { e.preventDefault(); if (i > 0) setSelId(ids[i-1]); }
      if (e.key >= '1' && e.key <= '5') {
        const n = parseInt(e.key);
        if (selId) handleSetStars(selId, sel?.stars === n ? 0 : n);
      }
      if (e.key === '0' && selId) handleSetStars(selId, 0);
      if ((e.key === 'x' || e.key === 'X') && selId) handleToggleUsedKb(selId);
      if (e.key === 'g' || e.key === 'G') setLoupeMode('grid');
      if ((e.key === 'e' || e.key === 'E') && isDone) setLoupeMode('loupe');
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [photos, selId, filteredPhotos, sel, handleSetStars, handleToggleUsedKb]);

  /* clear creative state when folder changes */
  useEffect(() => {
    setCreativeResults([]);
    setCreativeAnchor(null);
    setCreativePrompt('');
    setCreativeOutDir('');
    setCreativeFallback('');
    setCreativeSelection(null);
    setCreativeShowOriginal(false);
  }, [folder]);

  /* load photos when folder changes (skipped when resuming from catalog) */
  useEffect(() => {
    if (!folder.trim()) return;
    if (skipFolderLoadRef.current) { skipFolderLoadRef.current = false; return; }
    // Instant pre-grade niche recommendation — fire-and-forget in parallel with
    // the photo listing. Warm CPU CLIP on the server returns in <3 s; the picker
    // auto-selects the result and labels it "(Recommended)". Falls back silently
    // so a slow/failed detect never blocks folder loading.
    const safeFolder = sanitizePath(folder);
    setNicheRec(null);
    setNicheDetecting(true);
    axios.post(`${API}/api/recommend-niche`, { folder_path: safeFolder, folder_paths: [safeFolder] })
      .then(r => {
        if (r.data?.detected && r.data?.preset) { setNicheRec(r.data); setPreset(r.data.preset); }
      })
      .catch(() => {})
      .finally(() => setNicheDetecting(false));

    const load = async () => {
      setListLoading(true);
      try {
        const res = await axios.post(`${API}/api/list-folder`, { folder_path: sanitizePath(folder) });
        const rawPhotos: {path:string;exif:any}[] = res.data.photos || res.data.paths?.map((p: string) => ({path:p,exif:{}})) || [];
        if (!rawPhotos.length) notify("That folder has no images in it", "info");
        const ps = rawPhotos.map(p => ({ id:photoId(p.path), path:p.path, grade:'Pending', score:0, breakdown:{}, critique:'', reasoning_log:'', is_verified:false, stars:0, exif:p.exif||{} }));
        setPhotos(ps);
        setFolders([folder]);
        setSelId(ps[0]?.id ?? null);
        setMainTab('gallery');
        setLoupeMode('grid');
      } catch (err: any) { notify(`${err.response?.data?.detail || "Could not read that folder"}`, "error"); }
      finally { setListLoading(false); }
    };
    load();
  }, [folder]);

  /* load flags */
  useEffect(() => {
    axios.get(`${API}/api/flags/load`)
      .then(r => { setLocked(new Set(r.data.locked||[])); setUsed(new Set(r.data.used||[])); })
      .catch(() => {});
  }, []);

  /* check for saved catalog on first load */
  useEffect(() => {
    axios.get(`${API}/api/catalog?t=` + Date.now())
      .then(r => { if (r.data.exists && r.data.photos?.length) setCatalogBanner(true); })
      .catch(() => {});
  }, []);

  /* auto-save catalog (debounced 2s) whenever graded photos or folder list changes.
   * Failures are NOT silent: the first failure notifies the user and starts a
   * retry loop — a swallowed error here means Resume silently loses the session. */
  useEffect(() => {
    if (folders.length === 0 || !photos.some(p => p.grade !== 'Pending')) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      const photosToSave = photos.map(({ id: _id, ...rest }) => rest);
      axios.post(`${API}/api/catalog/save`, { photos: photosToSave, folders })
        .then(() => setCatalogSaveFailed(false))
        .catch(() => {
          setCatalogSaveFailed(prevFailed => {
            if (!prevFailed) notify('Changes are not being saved — your session will not resume after closing.', 'error');
            return true;
          });
        });
    }, 2000);
  }, [photos, folders, notify]);

  /* While saving is failing, retry every 15 s until it succeeds again. */
  useEffect(() => {
    if (!catalogSaveFailed) return;
    const id = setInterval(() => {
      axios.post(`${API}/api/catalog/save`, { photos: photos.map(({ id: _id, ...rest }) => rest), folders })
        .then(() => { setCatalogSaveFailed(false); notify('Saving resumed', 'success'); })
        .catch(() => {});
    }, 15_000);
    return () => clearInterval(id);
  }, [catalogSaveFailed, photos, folders, notify]);

  // Hide heatmap overlay whenever the selected photo changes
  useEffect(() => { setShowHeatmap(false); }, [selId]);

  // Reset jury critique state when selected photo changes
  useEffect(() => {
    setJuryCritique(null);
    setJuryThink(null);
    setJuryCritPath(null);
    setCritTrigger('');
    setIsAuditModeActive(false);
    setShowEyeOverlay(false);
    setPhotoNatDims(null);
    setDeepCritique(null);
    setDeepCritiqueLoading(false);
  }, [selId]);

  // Disable overlay when leaving the analysis tab; user toggles it on manually
  useEffect(() => {
    if (infoTab !== 'analysis') setIsAuditModeActive(false);
  }, [infoTab]);

  // Poll for annotation readiness when a photo is selected but not yet annotated
  useEffect(() => {
    if (!sel || (sel.has_annotations && sel.eye_overlay_url !== undefined)) return;
    let cancelled = false;
    let misses = 0;
    const check = async () => {
      if (cancelled || document.hidden) return;
      try {
        const stem = sel.path.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, '') ?? '';
        const r = await fetch(`${API}/api/annotations/${encodeURIComponent(stem)}`);
        if (!r.ok || cancelled) return;
        const d = await r.json();
        const gotAnnotations = d.has_annotations && d.score_factors?.length > 0;
        const gotOverlay     = Boolean(d.eye_overlay_url);
        if (gotAnnotations || gotOverlay) {
          setPhotos(prev => prev.map(p =>
            p.path === sel.path
              ? {
                  ...p,
                  ...(gotAnnotations ? {
                    has_annotations: d.has_annotations,
                    score_factors:   d.score_factors,
                  } : {}),
                  ...(gotOverlay ? { eye_overlay_url: d.eye_overlay_url } : {}),
                }
              : p
          ));
        } else if (++misses >= 15) {
          // 15 misses (~2 min): this photo is never getting annotations — stop
          // polling instead of hammering the endpoint every 8 s forever.
          cancelled = true;
        }
      } catch { /* silent */ }
    };
    const id = setInterval(check, 8000);
    const onVis = () => { if (!document.hidden) check(); };
    document.addEventListener('visibilitychange', onVis);
    check();
    return () => { cancelled = true; clearInterval(id); document.removeEventListener('visibilitychange', onVis); };
  }, [selId, sel?.has_annotations]);

  const toggleHeatmap = useCallback(async () => {
    if (!sel) return;
    if (showHeatmap) { setShowHeatmap(false); return; }
    if (heatmapPath === sel.path && heatmapB64) { setShowHeatmap(true); return; }
    setHeatmapLoading(true);
    try {
      const resp = await axios.get<{b64: string}>(
        `${API}/api/heatmap/technical/${encodeURIComponent(sel.path)}`
      );
      setHeatmapB64(resp.data.b64);
      setHeatmapPath(sel.path);
      setShowHeatmap(true);
    } catch { /* silent fail */ } finally {
      setHeatmapLoading(false);
    }
  }, [sel, showHeatmap, heatmapPath, heatmapB64]);

  /* critique trigger hover — lazy-loads heatmap when blur/heatmap type is hovered */
  const handleTriggerEnter = useCallback(async (type: string) => {
    setCritTrigger(type);
    if ((type === 'blur' || type === 'heatmap') && sel && (heatmapPath !== sel.path || !heatmapB64)) {
      setHeatmapLoading(true);
      try {
        const resp = await axios.get<{b64: string}>(
          `${API}/api/heatmap/technical/${encodeURIComponent(sel.path)}`
        );
        setHeatmapB64(resp.data.b64);
        setHeatmapPath(sel.path);
      } catch { /* silent — overlay just won't show */ } finally {
        setHeatmapLoading(false);
      }
    }
  }, [sel, heatmapPath, heatmapB64]);

  const handleTriggerLeave = useCallback(() => setCritTrigger(''), []);

  const handleRagUpload = useCallback(async (file: File) => {
    setRagUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      await axios.post(`${API}/api/rag/upload`, form, { timeout: 300000 });
      await fetchRagStatus();
      notify(`"${file.name}" ingested as reference`, 'success');
    } catch {
      notify('PDF ingestion failed — check server logs', 'error');
    } finally {
      setRagUploading(false);
    }
  }, [fetchRagStatus]);

  const handleRagClear = useCallback(async () => {
    try {
      await axios.delete(`${API}/api/rag/clear`);
      setRagPdfs([]);
      notify('Reference library cleared', 'info');
    } catch {
      notify('Clear failed', 'error');
    }
  }, []);

  const handlePegUpload = useCallback(async (file: File) => {
    setPegFile(file);
    setPegHash(null);
    setPegLoading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const resp = await axios.post<{status: string, hash: string}>(`${API}/api/upload`, form);
      setPegHash(resp.data.hash);
    } catch {
      notify('Reference upload failed', 'error');
      setPegFile(null);
    } finally {
      setPegLoading(false);
    }
  }, []);

  const handleSemanticSearch = useCallback(async (q: string) => {
    if (!q.trim()) { setSearchResults(null); return; }
    setSearchLoading(true);
    try {
      const resp = await axios.get<{results: {hash: string; path: string; score: number}[]}>(
        `${API}/api/search/semantic`, { params: { q } }
      );
      const paths = new Set(resp.data.results.map(r => r.path));
      setSearchResults(paths);
      if (paths.size === 0) notify(`No results for "${q}"`, 'info');
    } catch {
      notify('Semantic search failed', 'error');
    } finally {
      setSearchLoading(false);
    }
  }, [notify]);

  const handleJuryCritique = useCallback(async (path: string) => {
    if (!path) return;
    // Use the path stem as image_hash (MD5 stem used by the ingestion pipeline)
    const hash = path.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, '') ?? '';
    if (!hash) return;
    if (juryCritPath === path && juryCritique) return; // already loaded
    setJuryLoading(true);
    setJuryCritique(null);
    setJuryThink(null);
    try {
      const resp = await axios.get<{critique: string; think?: string; error?: string}>(
        `${API}/api/critique/jury/${encodeURIComponent(hash)}`
      );
      if (resp.data.error) {
        // Backend returned a structured error (GGUF missing, model crash, etc.)
        setJuryCritique(`Critique failed: ${resp.data.error}`);
        console.error('[jury] backend error:', resp.data.error);
      } else {
        setJuryCritique(resp.data.critique);
        setJuryThink(resp.data.think ?? null);
        setJuryCritPath(path);
        if (resp.data.think) console.debug('[jury <think>]', resp.data.think);
      }
    } catch (err: any) {
      // Network-level failure (server offline, timeout, etc.)
      const detail = err?.response?.data?.error ?? err?.response?.data?.detail ?? null;
      setJuryCritique(detail ? `Critique failed: ${detail}` : 'Server unreachable — is the backend running?');
      console.error('[jury] network error:', err);
    } finally {
      setJuryLoading(false);
    }
  }, [juryCritPath, juryCritique]);

  /* folder browser */
  const loadBrowser = useCallback(async (path: string) => {
    setBLoading(true);
    try {
      const r = await axios.post(`${API}/api/browse-folder`, { folder_path: path });
      setBFolders(r.data.folders || []);
      setBImages(r.data.images || []);
    } catch { } finally { setBLoading(false); }
  }, []);

  const goUp = useCallback(() => {
    const parts = bPath.replace(/[\\/]+$/, '').split(/[\\/]/).filter(Boolean);
    if (parts.length <= 1) { setBPath('C:\\'); loadBrowser('C:\\'); return; }
    parts.pop();
    const p = parts.join('\\') || 'C:\\';
    setBPath(p); loadBrowser(p);
  }, [bPath, loadBrowser]);

  const handleBrowserFolderClick = useCallback((e: MouseEvent, path: string, _idx: number) => {
    const isCtrl = (e as any).ctrlKey || (e as any).metaKey;
    if (isCtrl) {
      // Ctrl+click toggles folder selection (for multi-add)
      setBSelFolders(prev => {
        const next = new Set(prev);
        if (next.has(path)) next.delete(path); else next.add(path);
        return next;
      });
    } else {
      // Single click navigates into the folder
      setBPath(path);
      loadBrowser(path);
      setBSelFolders(new Set());
    }
  }, [loadBrowser]);

  const openBrowser    = useCallback(() => { setBrowserMode('open'); setShowBrowser(true); loadBrowser(bPath); }, [bPath, loadBrowser]);
  const openAddFolder  = useCallback(() => { setBrowserMode('add');  setShowBrowser(true); loadBrowser(bPath); }, [bPath, loadBrowser]);

  // Populate the gallery from a /api/catalog payload (graded-photo checkpoint).
  // Shared by Resume and by grade-failure recovery. Returns the photo count.
  const applyCatalog = useCallback((data: any): number => {
    if (!data?.exists || !data?.photos?.length) return 0;
    const ps = data.photos.map((p: any) => ({ ...p, id: photoId(p.path) }));
    // Apply same auto-redact logic as grading so duplicates are hidden in the gallery
    const autoRedacted = new Set<string>(
      ps.filter((p: any) => p.cluster_id >= 0 && !(p.sim_flag || '').startsWith('★'))
        .map((p: any) => p.path)
    );
    const firstVisible =
      ps.find((p: any) => !autoRedacted.has(p.path) && !((p.grade as string)?.includes('Weak'))) ??
      ps.find((p: any) => !autoRedacted.has(p.path));
    setPhotos(ps);
    setRedacted(autoRedacted);
    setSelId(firstVisible?.id ?? ps[0]?.id ?? null);
    return ps.length;
  }, []);

  const handleResume = useCallback(async () => {
    try {
      const r = await axios.get(`${API}/api/catalog?t=` + Date.now());
      if (!r.data.exists || !r.data.photos?.length) return;
      const savedFolders: string[] = r.data.folders || [];
      skipFolderLoadRef.current = true;
      setFolder(savedFolders[0] || '');
      setFolders(savedFolders);
      const n = applyCatalog(r.data);
      setLoupeMode('grid');
      setCatalogBanner(false);
      notify(`Resumed — ${n} photos from ${savedFolders.length} folder${savedFolders.length !== 1 ? 's' : ''}`, 'success');
    } catch { notify('Failed to resume session', 'error'); }
  }, [notify, applyCatalog]);

  const handleAddFolder = useCallback(async (newFolder: string) => {
    setListLoading(true);
    try {
      const res = await axios.post(`${API}/api/list-folder`, { folder_path: sanitizePath(newFolder) });
      const rawPhotos: {path:string;exif:any}[] = res.data.photos || [];
      setPhotos(prev => {
        const existing = new Set(prev.map(p => p.path));
        const added = rawPhotos
          .filter(p => !existing.has(p.path))
          .map(p => ({ id:photoId(p.path), path:p.path, grade:'Pending', score:0, breakdown:{}, critique:'', reasoning_log:'', is_verified:false, stars:0, exif:p.exif||{} }));
        return [...prev, ...added];
      });
      setFolders(prev => prev.includes(newFolder) ? prev : [...prev, newFolder]);
      notify(`Added ${rawPhotos.length} photos from ${newFolder.split(/[\\/]/).pop()}`, 'success');
    } catch { notify('Failed to add folder', 'error'); }
    finally { setListLoading(false); }
  }, [notify]);

  const pickFolder = useCallback(async () => {
    const pw = (window as any).pywebview;
    if (pw?.api?.pick_folder) {
      const p: string|null = await pw.api.pick_folder();
      if (p) { setFolder(p); setPhotos([]); setSelId(null); }
      return;
    }
    if (isTauri()) {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const s = await open({ directory:true, multiple:false, title:"Select Photo Folder" });
      if (typeof s === 'string' && s) { setFolder(s); setPhotos([]); setSelId(null); }
      return;
    }
    try {
      const r = await axios.get(`${API}/api/pick-folder`);
      if (r.data?.path) { setFolder(r.data.path); setPhotos([]); setSelId(null); }
    } catch { notify("Could not open the folder picker. Paste a folder path instead.", "error"); }
  }, [notify]);

  /* grade — uses SSE stream so large folders never time out */
  const handleGrade = useCallback(async (forceRescan = false, skipModal = false) => {
    const safePath = sanitizePath(folder);
    if (!safePath && folders.length === 0) { notify("Enter a folder path, or use Open folder to browse.", "error"); return; }
    if (!skipModal) {
      const photoCount = photos.length > 0 ? photos.length : 0;
      setPreGradeModal({ photoCount });
      // Kick off Vision Engine preload as soon as modal opens so it's warm by the time
      // the user clicks Start Culling.
      if (!graderStatus?.qwen_warm && !graderStatus?.qwen_loading) {
        axios.post(`${API}/api/models/preload`).catch(() => {});
      }
      // NOTE: pre-grade niche auto-detection via a scan pass is disabled — the
      // scan runs the full scoring pipeline (seconds-per-image) under gpu_lock,
      // which could not meet the <3s target and blocked previews. If nicheRec is
      // already known (from a prior grade), the picker still pre-selects + labels
      // it; otherwise the user picks manually. See _scan_folder_for_data / the
      // /api/recommend-niche endpoint for the (currently unused) scan path.
      return;
    }
    setLoading(true);
    setGradeProgress(0);
    setGradeDesc("");
    setGradeStartMs(Date.now());
    setGradeEtaSecs(null);
    const allFolderPaths = folders.length > 0 ? folders.map(sanitizePath) : [safePath];
    try {
      const resp = await fetch(`${API}/api/grade/v2/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: allFolderPaths[0], folder_paths: allFolderPaths, preset, scan_mode: scanMode, deep_grade: deepGrade, force_rescan: forceRescan, mogco_target: mogcoTarget }),
      });
      if (!resp.ok) {
        try { const d = await resp.json(); throw new Error(d.error ?? `Server error ${resp.status}`); }
        catch (e: any) { if (e.message && !e.message.startsWith('{')) throw e; throw new Error(`Server error ${resp.status}`); }
      }
      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      const _readWithTimeout = (): Promise<ReadableStreamReadResult<Uint8Array>> =>
        // The losing timer used to keep running after a successful read,
        // accumulating one dead 45 s timer per chunk on long grades.
        new Promise((resolve, reject) => {
          const t = setTimeout(
            () => reject(new Error('No response from server for 45 s — the grader may have crashed. Check the server log and click Grade to retry.')),
            45_000,
          );
          reader.read().then(
            v => { clearTimeout(t); resolve(v); },
            e => { clearTimeout(t); reject(e); },
          );
        });
      outer: while (true) {
        const { done, value } = await _readWithTimeout();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          let msg: any;
          try { msg = JSON.parse(line.slice(6)); } catch { continue; }
          if (msg.progress !== undefined) setGradeProgress(msg.progress);
          if (msg.desc) {
            // "Quality: Pro" is a one-off banner, not a stage — show it in the
            // badge rather than letting it flicker through the stage line.
            const q = /^Quality:\s*(.+)$/.exec(msg.desc);
            if (q) setGradeQuality(q[1].trim()); else setGradeDesc(msg.desc);
          }
          if (msg.quality) setGradeQuality(String(msg.quality));
          if (msg.error) throw new Error(msg.error);
          if (msg.done) {
            const ps = msg.data.map((p: any) => ({ ...p, id: photoId(p.path) }));
            setPhotos(ps);
            setRedacted(new Set<string>(
              ps.filter((p: any) => p.cluster_id >= 0 && !(p.sim_flag || '').startsWith('★'))
                .map((p: any) => p.path)
            ));
            const firstVisible = ps.find((p: any) => !((p.grade as string)?.includes('Weak')))
              ?? ps[0];
            setSelId(firstVisible?.id ?? ps[0]?.id ?? null);
            // Populate carousel from MOGCO result if present, else clear it
            if (msg.mogco_sequence?.length > 0) {
              setCarousel(msg.mogco_sequence);
              setSubjType('mogco-beam');
            } else {
              setCarousel([]);
            }
            setMainTab('gallery');
            setLoupeMode('loupe');
            setInfoTab('breakdown');
            setLoading(false);
            setGradeProgress(0);
            setGradeDesc("");
            const mogcoNote = msg.mogco_sequence?.length > 0
              ? ` · ${msg.mogco_sequence.length}-slot sequence ready`
              : '';
            const mogcoErr  = msg.mogco_error
              ? ` · Sequence: ${msg.mogco_error}`
              : '';
            if (msg.mogco_error) notify(`${msg.mogco_error}`, 'error');
            // Transparency: report which grader actually ran, and warn (don't hide)
            // when Deep Grade was requested but silently fell back to Fast (SigLIP)
            // because there wasn't enough free RAM to load the vision model.
            const _graders = new Set<string>(ps.map((p: any) => p?.breakdown?._grader).filter(Boolean));
            const _usedDeep = _graders.has('qwen');
            setGraderUsed(scanMode ? 'scan' : _usedDeep ? 'deep' : 'fast');
            if (deepGrade && !scanMode && !_usedDeep) {
              notify('Deep Grade fell back to Fast — not enough free memory for the deep analysis. Close some apps and re-grade for full accuracy.', 'error');
            }
            notify(`Graded ${msg.total} images${mogcoNote}${mogcoErr}`, 'success');
            axios.post(`${API}/api/recommend`, { photos: msg.data })
              .then(rec => setNicheRec(rec.data))
              .catch(() => {});
            break outer;
          }
        }
      }
    } catch (err: any) {
      // The grade stream failed — either the worker died (server emits an
      // explicit error) or the client read timed out (grader stalled). The
      // worker checkpoints completed grades to catalog.json before any crash,
      // so try to recover graded-so-far instead of dropping the user on a blank
      // gallery with a bare error.
      const msg = err?.message || 'Failed';
      const isStall = /No response from server/i.test(msg);
      try {
        const r = await axios.get(`${API}/api/catalog?t=` + Date.now());
        const n = applyCatalog(r.data);
        if (n > 0) {
          setMainTab('gallery');
          setLoupeMode('grid');
          notify(`Grading stopped early — recovered ${n} graded photo${n !== 1 ? 's' : ''} from the last checkpoint.`, 'error');
        } else {
          notify(isStall ? 'No response from the grader — it may be stalled. Check the server log and retry.' : `${msg}`, 'error');
        }
      } catch {
        notify(`${msg}`, 'error');
      }
    }
    setLoading(false);
    setGradeProgress(0);
  }, [folder, folders, preset, notify, applyCatalog]);

  /* generate sequence */
  const handleGenerate = useCallback(async () => {
    const pool = photos
      .filter(p => p.grade !== 'Pending')
      .filter(p => seqMinStars === 0 || (p.stars ?? 0) >= seqMinStars);
    const filterNote = seqMinStars > 0 ? ` rated ${seqMinStars}★+` : '';
    if (pool.length < 5) { notify(`A sequence needs at least 5 graded photos${filterNote}. Grade more, or clear the filter.`, 'error'); return; }
    setLoading(true);
    try {
      const res = await axios.post(`${API}/api/generate`, { photos: pool, seed: Math.floor(Math.random()*999999), avoid_paths: carousel.map((c: any) => c.path) });
      const d = res.data;
      setCarousel(Array.isArray(d) ? d : d.sequence);
      setSubjType(d.subject_type ?? null);
      setMainTab('gallery');
      notify('Sequence generated', 'success');
    } catch (err: any) { notify(`${err.response?.data?.detail || "Could not build the sequence"}`, "error"); }
    setLoading(false);
  }, [photos, carousel, notify, seqMinStars]);

  const handleExport = async () => {
    if (carousel.length < 5) return;
    try {
      const r = await axios.post(`${API}/api/editorial?fmt=portrait`, {
        photos: carousel.map(c => ({ path:c.path, grade:c.grade, score:c.score, breakdown:c.breakdown||{} })),
        excluded_paths: [],
      });
      const zip = r.data[0]?.zip;
      if (zip) { const a = document.createElement('a'); a.href = photoUrl(zip); a.download = 'Editorial_Carousel.zip'; a.click(); }
    } catch { notify('Export failed', 'error'); }
  };

  const handleSave = async () => {
    if (!carousel.length) return;
    const name = `Story ${saved.length + 1}`;
    try {
      await axios.post(`${API}/api/save-sequence`, { name, sequence: carousel });
      setSaved(prev => [...prev, { name, sequence: carousel }]);
      notify(`Saved as "${name}"`, 'success');
    } catch (err: any) { notify(`${err.response?.data?.detail || "Could not save the sequence"}`, 'error'); }
  };

  const handleDeleteSaved = useCallback((idx: number) => {
    setSaved(prev => prev.filter((_, i) => i !== idx));
  }, []);

  const handleRunCreativeDirection = useCallback(async () => {
    if (photos.length === 0) { notify('No photos loaded.', 'error'); return; }
    setCreativeLoading(true);
    setCreativeProgress(0);
    setCreativeStage('Initialising…');
    setCreativeResults([]);
    try {
      const resp = await fetch(`${API}/api/creative-direction/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          anchor_path:     creativeAnchor ? sanitizePath(creativeAnchor) : '',
          folder_path:     sanitizePath(folders[0] || folder),
          style_prompt:    creativePrompt,
          structure_mode:  creativeMode,
          n_target:        creativeCount,
          peg_image_hash:  pegHash ?? null,
          mode:            seqMode === 'auto' || seqMode === 'competition' ? seqMode : 'story',
        }),
      });
      if (!resp.ok) throw new Error(`Server error ${resp.status}`);
      const reader  = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      outer: while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          let msg: any;
          try { msg = JSON.parse(line.slice(6)); } catch { continue; }
          if (msg.progress !== undefined) setCreativeProgress(msg.progress);
          if (msg.desc)                   setCreativeStage(msg.desc);
          if (msg.error) throw new Error(msg.error);
          if (msg.done) {
            if (msg.data?.error) throw new Error(msg.data.error);
            const outputs = msg.data?.outputs ?? [];
            setCreativeResults(outputs);
            setCreativeOutDir(msg.data?.output_dir ?? '');
            setCreativeFallback(msg.data?.director_fallback ?? '');
            setCreativeSelection(msg.data?.selection ?? null);
            const ok = outputs.filter((r: any) => r.success).length;
            if (ok === 0 && outputs.length === 0) {
              notify('Creative Direction ran but produced no outputs.', 'info');
            } else if (msg.data?.director_fallback) {
              notify(`Picked the ${ok} highest-scoring photos — no art direction ran`, 'info');
            } else {
              notify(`Styled ${ok} of ${outputs.length} photos`, 'success');
            }
            break outer;
          }
        }
      }
    } catch (err: any) {
      notify(`Could not style the photos. ${err.message || err}`, 'error');
    } finally {
      setCreativeLoading(false);
      setCreativeProgress(0);
      setCreativeStage('');
    }
  }, [creativeAnchor, creativePrompt, creativeMode, creativeCount, photos, folder, folders, notify]);

  const handleSaveSequence = useCallback(async () => {
    const successes = creativeResults.filter((r: any) => r.success);
    if (!successes.length) return;
    setSequenceSaving(true);
    try {
      const resp = await fetch(`${API}/api/creative-direction/save-sequence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ outputs: successes, base_dir: creativeOutDir }),
      });
      const data = await resp.json();
      if (data.ok) {
        notify(`Saved ${data.count} photos to ${data.story_dir.split(/[\\/]/).pop()}`, 'success');
        setUsedCount(data.used_total ?? 0);
      } else {
        notify(`Could not save. ${data.error}`, 'error');
      }
    } catch (err: any) {
      notify(`Could not save. ${err.message}`, 'error');
    } finally {
      setSequenceSaving(false);
    }
  }, [creativeResults, creativeOutDir, notify]);

  const handleClearUsed = useCallback(async () => {
    try {
      const resp = await fetch(`${API}/api/creative-direction/clear-used`, { method: 'POST' });
      const data = await resp.json();
      if (data.ok) {
        setUsedCount(0);
        notify('History cleared — all photos eligible again', 'success');
      }
    } catch (err: any) {
      notify(`Could not clear. ${err.message}`, 'error');
    }
  }, [notify]);

  const handleSortByStars = useCallback((n: number) => {
    setCarousel(prev => [...prev].sort((a, b) => {
      const aS = a.stars ?? 0, bS = b.stars ?? 0;
      // Exact match to chosen star level floats to top, then descending
      const aMatch = aS === n ? 1 : 0, bMatch = bS === n ? 1 : 0;
      return bMatch !== aMatch ? bMatch - aMatch : bS - aS;
    }));
  }, []);

  const toggleFlag = useCallback(async (path: string, type: 'lock'|'used') => {
    try {
      await axios.post(`${API}/api/flags/${type}`, { path });
      const setter = type === 'lock' ? setLocked : setUsed;
      setter(prev => { const n = new Set(prev); n.has(path) ? n.delete(path) : n.add(path); return n; });
    } catch (err: any) { notify(`${err.response?.data?.detail || `Failed to toggle ${type}`}`, 'error'); }
  }, [notify]);

  const handleDragEnd = (e: any) => {
    // `over` is null when the drop lands outside every droppable — dnd-kit
    // still fires onDragEnd. Guard first or the reorder below throws.
    if (!e.over || e.active.id === e.over.id) return;
    setCarousel(prev => {
      const a = [...prev];
      const oi = a.findIndex(i => i.path === e.active.id);
      const ni = a.findIndex(i => i.path === e.over.id);
      if (oi === -1 || ni === -1) return prev;
      const [m] = a.splice(oi, 1); a.splice(ni, 0, m);
      return a;
    });
  };

  const handleCopyPath = useCallback(() => {
    if (!sel?.path) return;
    navigator.clipboard.writeText(sel.path);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }, [sel]);

  const jumpToPhoto = useCallback((path: string) => {
    const p = photos.find(ph => ph.path === path);
    if (p) { setSelId(p.id); setMainTab('gallery'); }
  }, [photos]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    dragCounter.current = 0;
    setDragOver(false);
    const item = e.dataTransfer.items?.[0];
    const file = e.dataTransfer.files?.[0];
    // `File.path` is an ELECTRON extension. This app runs in pywebview /
    // WebView2, where a File carries no filesystem path at all — so this was
    // always undefined and the handler returned here, every single time.
    // Dropping a folder never did anything, silently, and looked like the app
    // ignoring the gesture.
    //
    // A webview genuinely cannot resolve a dropped folder to a path: the HTML
    // File API deliberately does not expose one. The honest response is to
    // open the native picker the drop was trying to shortcut, so the gesture
    // still gets the user where they were going.
    const fullPath = (file as any)?.path as string | undefined;
    if (!fullPath) {
      openBrowser();
      return;
    }
    const entry = item?.webkitGetAsEntry?.();
    const isDir = entry?.isDirectory || fullPath.endsWith('/') || fullPath.endsWith('\\');
    const fp = isDir ? fullPath : fullPath.split(/[\\/]/).slice(0, -1).join('/') || fullPath;
    if (fp) { setFolder(fp); setPhotos([]); setSelId(null); }
  }, [openBrowser]);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current++;
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current--;
    if (dragCounter.current === 0) setDragOver(false);
  }, []);

  const handleCreateFromSelection = useCallback(() => {
    if (!selectedIds.size) { notify('Select photos first', 'error'); return; }
    const sel = photos.filter(p => selectedIds.has(p.id));
    setCarousel(sel);
    setSelectedIds(new Set());
    setSelectMode(false);
    setMainTab('sequence');
    notify('Sequence created from selection', 'success');
  }, [photos, selectedIds, notify]);

  const onResizeDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const sx = e.clientX, sw = rightW;
    const onMove = (ev: MouseEvent) => setRightW(Math.max(200, Math.min(460, sw - (ev.clientX - sx))));
    const onUp = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
    window.addEventListener('mousemove', onMove); window.addEventListener('mouseup', onUp);
  }, [rightW]);

  const isGrading = loading;
  const isDone    = !loading && photos.length > 0 && photos.some(p => p.grade !== 'Pending');
  // If grading is reset/cleared, don't stay on a post-grade tab
  useEffect(() => {
    if (!isDone && mainTab !== 'gallery') setMainTab('gallery');
  }, [isDone, mainTab]);
  const picks     = photos.filter(p => gradeLabel(p.grade) === 'Strong' && !redacted.has(p.path)).length;
  const mids      = photos.filter(p => gradeLabel(p.grade) === 'Mid'    && !redacted.has(p.path)).length;
  // Paths marked as used: server flags + photos committed to any saved sequence
  const allUsedPaths = useMemo(() =>
    new Set([...Array.from(used), ...saved.flatMap(s => s.sequence.map((p: any) => p.path))]),
  [used, saved]);
  const rejects   = photos.filter(p => gradeLabel(p.grade) === 'Weak'    && !redacted.has(p.path)).length;
  // Star counts within the current grade filter (for the filter bar labels)
  const gradeFiltered = filterGrade ? photos.filter(p => gradeLabel(p.grade) === filterGrade) : photos;
  const starCounts = [0,1,2,3,4,5].map(n =>
    n === 0 ? gradeFiltered.filter(p => !p.stars).length
            : gradeFiltered.filter(p => p.stars === n).length
  );
  const selIdx    = filteredPhotos.findIndex(p => p.id === selId);
  const hasPrev   = selIdx > 0;
  const hasNext   = selIdx < filteredPhotos.length - 1;
  const isGraded  = isDone && sel && sel.grade !== 'Pending';

  const sequenceNarrative = useMemo(() => {
    if (!carousel.length) return null;
    const LMAP: Record<string,string> = {
      "Technical":"tech","News Sharpness":"tech","Cleanliness":"tech","Execution":"tech",
      "Detail Retention":"tech","Exposure":"tech","Sharpness & Detail":"tech",
      "Composition":"comp","Framing":"comp","Context":"comp","Geometry & Balance":"comp",
      "Negative Space":"comp","Framing Instinct":"comp","Layered Depth":"comp",
      "Lighting":"light","Atmosphere":"light","Natural Light":"light","Mood & Tone":"light",
      "Tonal Purity":"light","Contrast Purity":"light","Available Light":"light",
      "Natural Light Quality":"light",
      "Decisive Moment":"auth","Cultural Depth":"auth","Journalistic Integrity":"auth",
      "Narrative Suggestion":"auth","Conceptual Weight":"auth","Reduction":"auth",
      "Authenticity":"auth","Immediacy":"auth","Environmental Truth":"auth",
      "Subject Isolation":"human","Sense of Place":"human","Human Impact":"human",
      "Character Presence":"human","Emotional Resonance":"human","Scale Element":"human",
      "Human/Culture":"human","Presence":"human","Scale & Life":"human",
    };
    const tot: Record<string,number> = {tech:0,comp:0,light:0,auth:0,human:0};
    const cnt: Record<string,number> = {tech:0,comp:0,light:0,auth:0,human:0};
    carousel.forEach(c => {
      Object.entries(c.breakdown || {}).forEach(([lbl, val]) => {
        const k = LMAP[lbl];
        if (k && typeof val === 'number') { tot[k] += val; cnt[k]++; }
      });
    });
    const dimKeys = ['tech','comp','light','auth','human'] as const;
    const avg: Record<string,number> = {};
    dimKeys.forEach(k => { avg[k] = cnt[k] ? tot[k]/cnt[k] : 0; });
    const sorted = [...dimKeys].sort((a,b) => avg[b]-avg[a]);
    const strongest = sorted[0], weakest = sorted[sorted.length-1];
    const dimLabels: Record<string,string> = {
      tech:  'technical precision', comp: 'compositional instinct',
      light: 'atmospheric light',  auth: 'decisive-moment capture',
      human: 'human presence',
    };
    const nicheCtx: Record<string,string> = {
      'Classic Street':       'The sequence carries documentary hallmarks: authentic gesture, layered framing, and a sense of life caught mid-breath.',
      'Travel Editor':         'The sequence reads like a dispatched edit — cultural immersion, sense of place, and subjects genuinely encountered rather than posed.',
      'Photojournalism':       'The sequence holds documentary weight: technically grounded, contextually honest, anchored in authentic human stakes.',
      'Cinematic/Editorial':   'Light is the connective tissue. The sequence moves through moods rather than subjects — each frame builds atmosphere for the next.',
      'Fine Art/Contemporary': 'The sequence operates conceptually — compositional logic over candid impulse, tonal control over spontaneous capture.',
      'Minimalist/Urbex':      'Structure drives the sequence. Negative space and geometric restraint create rhythm without relying on human narrative.',
      'London Street':         'The sequence has the quality of a slow walk through a city at dusk — atmospheric, human, unhurried.',
      'Humanist/Everyday':     'The sequence is rooted in people. Dignity, proximity, and warmth thread through each frame.',
      'Landscape with Elements':'Light and environment carry the weight. The sequence breathes through its landscapes — foreground, depth, and tonal gradation.',
      'Snapshot / Point-and-Shoot': 'The sequence has the energy of unfiltered presence — raw, immediate, unconcerned with perfection.',
    };
    const niche = nicheRec?.preset ?? preset;
    const ctx   = nicheCtx[niche] ?? nicheRec?.reason ?? '';
    const parts: string[] = [];
    parts.push(`${carousel.length}-frame sequence evaluated against ${niche}.`);
    if (ctx) parts.push(ctx);
    if (avg[strongest] > 0) parts.push(`Dominant quality across the edit: ${dimLabels[strongest]}.`);
    if (weakest !== strongest && avg[weakest] > 0) parts.push(`Area with most room to grow: ${dimLabels[weakest]}.`);
    return parts.join(' ');
  }, [carousel, nicheRec, preset]);

  if (!backendReady) {
    return (
      <div className="fixed inset-0 flex flex-col items-center justify-center gap-4 bg-ground">
        {backendError ? (
          <>
            {/* An error explains what happened and what to do next. The previous
                version showed a ⚠️ over "Make sure the app is running correctly",
                which tells someone staring at a stopped app precisely nothing. */}
            <p className="t-label !text-alarm-crit">Not connected</p>
            <p className="max-w-[38ch] text-center text-sm text-ink">
              FrameGrade can't reach its engine, so nothing can be graded yet.
            </p>
            <p className="max-w-[42ch] text-center text-xs text-ink-3">
              It usually means the engine is still starting. Give it a few seconds and retry —
              if it keeps failing, close the app completely and open it again.
            </p>
            <Button className="mt-2" variant="solid"
              onClick={() => { setBackendError(false); window.location.reload(); }}>
              Retry
            </Button>
          </>
        ) : (
          <>
            <div style={{ width:40, height:40, border:`3px solid ${T.raisedHover}`, borderTopColor:T.ink3, borderRadius:'var(--r-round)', animation:'spin .8s linear infinite' }}/>
            <span style={{ fontSize:'var(--text-sm)', color:T.ink2, letterSpacing:'var(--track-body)' }}>Starting FrameGrade…</span>
          </>
        )}
      </div>
    );
  }

  return (
    <div
      style={{ display:'flex', flexDirection:'column', height:'100vh', background:T.ground, overflow:'hidden',
        fontFamily:"var(--font-sans)", fontSize:'var(--text-md)', color:T.ink }}
      onDrop={handleDrop} onDragOver={e => { e.preventDefault(); e.stopPropagation(); }} onDragEnter={handleDragEnter} onDragLeave={handleDragLeave}
    >

      {/* Drag-and-drop overlay */}
      {dragOver && (
        <div style={{ position:'fixed', inset:8, zIndex:200, pointerEvents:'none', borderRadius:'var(--r-md)',
          display:'flex', alignItems:'center', justifyContent:'center',
          background:T.scrim, backdropFilter:'blur(6px)',
          border:`2px dashed ${T.ink3}`,
        }}>
          <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:12 }}>
            <FolderOpen size={48} strokeWidth={1} style={{ color:T.ink2 }}/>
            <span className="text-md text-ink">Drop folder to load</span>
          </div>
        </div>
      )}

      {/* Toast */}
      {toast && (
        <div style={{
          position:'fixed', top:12, left:'50%', transform:'translateX(-50%)', zIndex:300,
          padding:'7px 16px', borderRadius:'var(--r-md)', fontSize:'var(--text-sm)', fontWeight:500, whiteSpace:'nowrap',
          // Success is silence: a completed action needs no green. Only an error
          // takes a hue, and only on the border — "everything is fine" is
          // expressed by the absence of colour, so colour here always means act.
          background: T.raised,
          border:`1px solid ${toast.type==='error' ? T.alarmCrit : T.lineStrong}`,
          color:T.ink, animation:'slideUp .3s cubic-bezier(.2,0,0,1)',
        }}>{toast.msg}</div>
      )}

      {/* Engine health warning banner */}
      {engineHealth.status !== "checking" && (() => {
        // One list. This used to split on !endsWith('.gguf') because in the
        // Ollama era those were Ollama tags; the registry now offers nothing
        // but .gguf, so the "ollama" half was always empty -- and it gated the
        // Download button, which therefore never rendered.
        const missingModels = engineHealth.missing ?? [];
        const isOffline     = engineHealth.status === "offline";

        // Model load state chips — gemma3:4b and qwen2.5vl:3b
        // Empty by design. These chips reported gpu/cpu residency per model,
        // but /api/ollama/status sets size_vram to 0 for every entry -- with
        // llama_cpp the offload level is decided per call, not held as a
        // server-side fact. The chips could only ever show "absent", for models
        // (gemma3:4b, qwen2.5vl:3b) this app stopped shipping when Ollama was
        // removed. The JSX below is kept but never renders.
        const modelChips: {label:string; display:string; state:"gpu"|"cpu"|"absent"; size_vram?:number}[] = [];
        const anyCpu = false, anyAbsent = false;   // no residency signal exists
        const showBanner = isOffline || missingModels.length > 0;
        if (!showBanner) return null;
        if (bannerDismissed && !isOffline) return null;

        return (
          <div style={{
            position:'fixed', top:0, left:0, right:0, zIndex:250,
            padding: isOffline ? '10px 18px' : '7px 16px',
            fontSize:'var(--text-sm)', fontWeight: isOffline ? 600 : 500,
            // Offline is the one banner the user must act on, so it takes the
            // alarm fill. A missing optional model is informational: neutral
            // surface, with the alarm colour reduced to a single hairline.
            background: isOffline ? T.alarmWarn : T.surface,
            borderBottom: isOffline ? `2px solid ${T.alarmCrit}` : `1px solid ${T.alarmWarn}`,
            color: isOffline ? T.well : T.ink2,
            display:'flex', alignItems:'center', gap:10, flexWrap:'wrap',
          }}>
            <span className="h-1 w-1 shrink-0 rounded-full" style={{ background: 'currentColor' }}/>
            {isOffline ? (
              <span style={{ flex:1, minWidth:0 }}>
                <strong>The writing engine isn't running.</strong>{' '}
                Grading still works. Written critiques, the creative director, and photo
                annotations need a local model installed.
              </span>
            ) : (
              <span style={{ flex:1, minWidth:0 }}>
                {missingModels.length > 0 && <>Optional model{missingModels.length > 1 ? "s" : ""} not installed: <strong>{missingModels.join(", ")}</strong> — grading works without {missingModels.length > 1 ? "them" : "it"}; the writing features need {missingModels.length > 1 ? "them" : "it"}.</>}
              </span>
            )}
            {/* Ollama out-of-date — overrides all other controls */}
            {updateRequired ? (
              <>
                <span style={{ flex:1, minWidth:0, fontSize:'var(--text-sm)', fontWeight:600, color:T.alarmCrit }}>
                  Your Ollama engine is out of date and cannot run these models.
                </span>
                <a
                  href="https://ollama.com/download"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ flexShrink:0, padding:'4px 12px', fontSize:'var(--text-sm)', fontWeight:600,
                    background:T.raised, color:T.ink,
                    border:`1px solid ${T.lineStrong}`, borderRadius:'var(--r-md)', cursor:'pointer',
                    whiteSpace:'nowrap', textDecoration:'none' }}>
                  Download Ollama Update
                </a>
              </>
            ) : (
              <>
                {/* Generic error */}
                {downloadError && !isDownloading && (
                  <span style={{ fontSize:'var(--text-xs)', color:T.alarmCrit, fontWeight:600, flex:1, minWidth:0 }}>
                    ✕ {downloadError}
                  </span>
                )}
                {/* Download / Retry button */}
                {missingModels.length > 0 && !isDownloading && (
                  <button
                    onClick={() => { setDownloadError(null); handleDownloadMissing(); }}
                    style={{ flexShrink:0, padding:'4px 12px', fontSize:'var(--text-sm)', fontWeight:600,
                      background: T.raised,
                      color:T.ink, border:`1px solid ${downloadError ? T.alarmCrit : T.lineStrong}`,
                      borderRadius:'var(--r-md)', cursor:'pointer', whiteSpace:'nowrap' }}>
                    {downloadError ? 'Retry Download' : 'Download Missing Models'}
                  </button>
                )}
              </>
            )}
            {/* VLM model load state chips */}
            {!isOffline && (anyCpu || anyAbsent) && (
              <div style={{ display:'flex', alignItems:'center', gap:6, flexShrink:0, flexWrap:'wrap' }}>
                {modelChips.map(chip => {
                  const isGpu = chip.state === "gpu";
                  const isCpu = chip.state === "cpu";
                  return (
                    <span key={chip.label} title={isGpu ? `${chip.display} loaded in VRAM` : isCpu ? `${chip.display} running on CPU — VRAM headroom low` : `${chip.display} not loaded`}
                      style={{
                        padding:'2px 8px', borderRadius:'var(--r-sm)', fontSize:'var(--text-xs)', fontWeight:600, whiteSpace:'nowrap',
                        // A model sitting in VRAM is the expected case, so it is
                        // silent. Only CPU fallback — the case with a real cost
                        // the user can act on — reaches for the alarm token.
                        background: T.raised,
                        border: `1px solid ${isCpu ? T.alarmWarn : T.lineStrong}`,
                        color: isGpu ? T.ink2 : isCpu ? T.alarmWarn : T.ink3,
                      }}>
                      {chip.display} {isGpu ? '✓ GPU' : isCpu ? '⚡ CPU' : '—'}
                    </span>
                  );
                })}
                {anyCpu && <span style={{ fontSize:'var(--text-xs)', color:T.alarmWarn, fontWeight:500 }}>VRAM pressure — inference may be slow</span>}
              </div>
            )}
            {/* Progress indicator while downloading */}
            {isDownloading && (
              <div style={{ flexShrink:0, display:'flex', alignItems:'center', gap:8 }}>
                <div style={{ width:120, height:6, background:T.raised, borderRadius:'var(--r-sm)', overflow:'hidden' }}>
                  <div style={{ width:`${downloadProgress}%`, height:'100%', background:T.ink3, borderRadius:'var(--r-sm)', transition:'width .3s ease' }}/>
                </div>
                <span style={{ fontSize:'var(--text-xs)', whiteSpace:'nowrap', color:T.ink2 }}>
                  {currentDownloadModel}: {downloadProgress}% — do not close the app
                </span>
              </div>
            )}
            {/* Dismiss button */}
            {!isOffline && !isDownloading && (
              <button
                onClick={() => setBannerDismissed(true)}
                title="Dismiss"
                style={{ marginLeft:'auto', flexShrink:0, background:'none', border:'none', cursor:'pointer',
                  color: isOffline ? T.well : T.ink3, fontSize:'var(--text-md)', lineHeight:'var(--leading-none)', padding:'2px 4px' }}>
                ✕
              </button>
            )}
          </div>
        );
      })()}

      {/* Pre-grade info modal */}
      {preGradeModal && (
        <div style={{ position:'fixed', inset:0, zIndex:400, background:T.well, display:'flex', alignItems:'center', justifyContent:'center' }}
          role="presentation"
          onClick={() => setPreGradeModal(null)}>
          <div ref={preGradeDialogRef} style={{ background:T.surface1, border:`1px solid ${T.lineStrong}`, borderRadius:'var(--r-md)', padding:'28px 32px', maxWidth:420, width:'90%', display:'flex', flexDirection:'column', gap:16 }}
            role="dialog" aria-modal="true" aria-labelledby="pregrade-title"
            onClick={e => e.stopPropagation()}>
            {/* Header: eyebrow + quiet photo count (moved out of the body so it
                doesn't float alone between sections) */}
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline' }}>
              <div className="t-label">Before you start</div>
              {preGradeModal.photoCount > 0 && (
                <span className="t-num" style={{ fontSize:'var(--text-xs)', color:T.ink3 }}>
                  {preGradeModal.photoCount.toLocaleString()} photo{preGradeModal.photoCount !== 1 ? 's' : ''}
                </span>
              )}
            </div>

            {/* Readiness — one quiet list instead of stacked cards. Expected
                states are plain rows; only states the user can act on (low
                memory, downloads) carry colour. */}
            <div style={{ display:'flex', flexDirection:'column' }}>
              {!graderStatus?.draft_available ? (
                <div style={{ display:'flex', gap:10, padding:'10px 0', borderBottom:`1px solid ${T.line}` }}>
                  <div style={{ width:6, height:6, borderRadius:'var(--r-round)', background:T.alarmWarn, flexShrink:0, marginTop:6 }}/>
                  <div style={{ flex:1, minWidth:0 }}>
                    <div style={{ fontSize:'var(--text-sm)', color:T.ink }}>
                      {graderStatus?.qwen_download_pct != null
                        ? `Downloading Vision Engine — ${graderStatus.qwen_download_pct}%`
                        : 'Vision Engine: downloading in background…'}
                    </div>
                    {graderStatus?.qwen_download_pct != null && (
                      <div style={{ height:3, background:T.raisedHover, borderRadius:'var(--r-sm)', overflow:'hidden', margin:'6px 0 4px' }}>
                        <div style={{ height:'100%', width:`${graderStatus.qwen_download_pct}%`,
                          background:T.ink3, transition:'width .8s cubic-bezier(.2,0,0,1)' }}/>
                      </div>
                    )}
                    <div style={{ fontSize:'var(--text-xs)', color:T.ink3, lineHeight:'var(--leading-body)', marginTop:2 }}>
                      ~6 GB one-time download, runs in the background. You can start grading now — it begins once complete.
                    </div>
                  </div>
                </div>
              ) : graderStatus?.qwen_warm ? (
                <div style={{ display:'flex', gap:10, padding:'10px 0', borderBottom:`1px solid ${T.line}` }}>
                  <div style={{ width:6, height:6, borderRadius:'var(--r-round)', background:T.ink3, flexShrink:0, marginTop:6 }}/>
                  <div style={{ flex:1, minWidth:0 }}>
                    <div style={{ fontSize:'var(--text-sm)', color:T.ink }}>Vision Engine ready</div>
                    <div style={{ fontSize:'var(--text-xs)', color:T.ink3, lineHeight:'var(--leading-body)', marginTop:2 }}>Loaded in VRAM — grading starts immediately.</div>
                  </div>
                </div>
              ) : graderStatus?.qwen_loading ? (
                <div style={{ display:'flex', gap:10, padding:'10px 0', borderBottom:`1px solid ${T.line}` }}>
                  <div style={{ width:9, height:9, borderRadius:'var(--r-round)', border:`2px solid ${T.ink3}`, borderTopColor:'transparent', animation:'spin .8s linear infinite', flexShrink:0, marginTop:4 }}/>
                  <div style={{ flex:1, minWidth:0 }}>
                    <div style={{ fontSize:'var(--text-sm)', color:T.ink }}>Loading Vision Engine…</div>
                    <div style={{ fontSize:'var(--text-xs)', color:T.ink3, lineHeight:'var(--leading-body)', marginTop:2 }}>~30–60 seconds — Start Culling unlocks automatically.</div>
                  </div>
                </div>
              ) : (
                <div style={{ display:'flex', gap:10, padding:'10px 0', borderBottom:`1px solid ${T.line}` }}>
                  <div style={{ width:6, height:6, borderRadius:'var(--r-round)', background:T.ink3, flexShrink:0, marginTop:6 }}/>
                  <div style={{ flex:1, minWidth:0 }}>
                    <div style={{ fontSize:'var(--text-sm)', color:T.ink }}>Vision Engine ready</div>
                    <div style={{ fontSize:'var(--text-xs)', color:T.ink3, lineHeight:'var(--leading-body)', marginTop:2 }}>Model cached on disk — first load takes ~30–60 seconds.</div>
                  </div>
                </div>
              )}

              {/* System-RAM readiness — is it clear to grade? (live, polled every 2 s).
                  clear / tight / critical maps onto quiet / warn / crit: "clear" is
                  the expected state and stays neutral; only the two states the user
                  can act on carry a colour. */}
              {(sysRam || graderStatus) && (() => {
                const r = ramReadiness(sysRam ?? graderStatus);
                if (r.level === 'unknown') return null;
                const row = {
                  clear:    { col:T.ink3,      text:`System memory clear — ${r.free?.toFixed(1)} GB free, plenty of headroom for a full cull.` },
                  tight:    { col:T.alarmWarn, text:`System memory tight — ${r.free?.toFixed(1)} GB free. Grading will run, but closing a few apps gives the best results.` },
                  critical: { col:T.alarmCrit, text:`Low system memory — only ${r.free?.toFixed(1)} GB free, below the ~${r.min} GB needed. Close some apps before grading.` },
                }[r.level]!;
                return (
                  <div style={{ display:'flex', gap:10, padding:'10px 0', borderBottom:`1px solid ${T.line}` }}>
                    <div style={{ width:6, height:6, borderRadius:'var(--r-round)', background:row.col, flexShrink:0, marginTop:6 }}/>
                    <div style={{ flex:1, minWidth:0, fontSize:'var(--text-sm)', lineHeight:'var(--leading-body)',
                      color: r.level === 'clear' ? T.ink2 : row.col }}>{row.text}</div>
                  </div>
                );
              })()}

              {/* One-time INT4 quantisation disclaimer — only until the
                  pre-quantised cache exists on disk */}
              {graderStatus?.draft_available && !graderStatus?.qwen_int4_cached && !graderStatus?.qwen_warm && (
                <div style={{ display:'flex', gap:10, padding:'10px 0', borderBottom:`1px solid ${T.line}` }}>
                  <div style={{ width:6, height:6, borderRadius:'var(--r-round)', background:T.ink3, flexShrink:0, marginTop:6 }}/>
                  <div style={{ flex:1, minWidth:0 }}>
                    <div style={{ fontSize:'var(--text-sm)', color:T.ink }}>First cull: one-time engine optimisation</div>
                    <div style={{ fontSize:'var(--text-xs)', color:T.ink3, lineHeight:'var(--leading-body)', marginTop:2 }}>
                      The engine compresses for your GPU on this run — expect a pause of a few minutes around 52%. Saved afterwards, so every later cull skips it.
                    </div>
                  </div>
                </div>
              )}

              {/* Pipeline calibration warmup status */}
              {graderStatus?.warmup_running && (
                <div style={{ display:'flex', gap:10, padding:'10px 0', borderBottom:`1px solid ${T.line}` }}>
                  <div style={{ width:9, height:9, borderRadius:'var(--r-round)', border:`2px solid ${T.ink3}`, borderTopColor:'transparent', animation:'spin .8s linear infinite', flexShrink:0, marginTop:4 }}/>
                  <div style={{ flex:1, minWidth:0 }}>
                    <div style={{ fontSize:'var(--text-sm)', color:T.ink }}>Calibrating pipeline…</div>
                    <div style={{ fontSize:'var(--text-xs)', color:T.ink3, lineHeight:'var(--leading-body)', marginTop:2 }}>Warming CUDA kernels on your best photos — Start Culling unlocks when done.</div>
                  </div>
                </div>
              )}
              {graderStatus?.warmup_done && !graderStatus?.warmup_running && (
                <div style={{ display:'flex', gap:10, padding:'10px 0', borderBottom:`1px solid ${T.line}` }}>
                  <div style={{ width:6, height:6, borderRadius:'var(--r-round)', background:T.ink3, flexShrink:0, marginTop:6 }}/>
                  <div style={{ flex:1, minWidth:0, fontSize:'var(--text-sm)', color:T.ink3, lineHeight:'var(--leading-body)' }}>
                    Pipeline calibrated — first cull of this session will be fast.
                  </div>
                </div>
              )}
            </div>

            {/* Options — re-grade scope + niche, one consistent label/value rhythm */}
            <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
              {/* Re-grade scope */}
              <div>
                <div className="t-label" style={{ marginBottom:6 }}>Re-grade</div>
                <Segmented
                  value={rescanAll ? 'all' : 'new'}
                  onChange={v => setRescanAll(v === 'all')}
                  options={[
                    { value: 'all', label: 'Re-grade everything' },
                    { value: 'new', label: 'New photos only' },
                  ]}
                />
                <p className="text-xs text-ink-3" style={{ marginTop:6 }}>
                  {rescanAll
                    ? 'Every photo runs through the full pipeline.'
                    : 'Already-graded photos are skipped — only new additions are scored.'}
                </p>
              </div>

              {/* Niche picker */}
              <div>
                <div className="t-label flex items-center" style={{ marginBottom:6 }}>
                  <span>Photography niche</span>
                  {nicheDetecting && (
                    <span className="flex items-center gap-1 normal-case tracking-normal text-ink-3" style={{ marginLeft:'auto' }}>
                      <span style={{ width:9, height:9, borderRadius:'var(--r-round)', border:'2px solid currentColor', borderTopColor:'transparent', animation:'spin .8s linear infinite' }}/>
                      Detecting ideal niche…
                    </span>
                  )}
                  {!nicheDetecting && nicheRec?.detected && nicheRec?.preset === preset && (
                    <span className="normal-case tracking-normal text-ink-3" style={{ marginLeft:'auto' }}>
                      auto-selected
                    </span>
                  )}
                </div>
                <select
                  value={preset}
                  aria-label="Grading niche / preset"
                  onChange={e => setPreset(e.target.value)}
                  className="w-full cursor-pointer rounded-sm border border-line-strong bg-raised px-2 py-1 text-sm text-ink outline-none focus-visible:border-focus">
                  {NICHE_GROUPS.map(g => (
                    <optgroup key={g.category} label={g.category}>
                      {g.niches.map(n => (
                        <option key={n.key} value={n.key}>
                          {n.label}{nicheRec?.preset === n.key ? '  (Recommended)' : ''}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              </div>
            </div>

            {/* Deep Grade toggle — default OFF = fast SigLIP zero-shot; ON = Qwen VLM.
                Unboxed: a plain row separated by a hairline, long detail on hover. */}
            <label className="flex cursor-pointer items-start gap-2"
              style={{ paddingTop:12, borderTop:`1px solid ${T.line}` }}
              title="Off: fast grading — light on memory, recommended. On: each photo is read in detail (more nuanced, slower, heavier on memory).">
              <input type="checkbox" checked={deepGrade} onChange={e => setDeepGrade(e.target.checked)}
                className="mt-px cursor-pointer" style={{ accentColor: T.ink }} />
              <div>
                <div className="text-sm text-ink">Deep grade</div>
                <div className="text-xs text-ink-3" style={{ marginTop:2 }}>
                  Off: fast and light on memory (recommended). On: each photo is read in detail — slower, heavier.
                </div>
              </div>
            </label>

            {/* Actions */}
            {(() => {
              const _notReady = graderStatus?.qwen_loading || graderStatus?.qwen_download_pct != null || graderStatus?.warmup_running;
              return (
                <div className="mt-1 flex justify-end gap-2">
                  <Button onClick={() => setPreGradeModal(null)}>
                    Cancel
                  </Button>
                  {/* The confirm was the accent-filled "primary" this design does
                      not have: emphasis is luminance, so it is simply the solid
                      variant sitting next to a bordered one. */}
                  <Button
                    variant="solid"
                    disabled={!!_notReady}
                    autoFocus
                    onClick={() => { setPreGradeModal(null); handleGrade(rescanAll, true); }}
                    icon={(graderStatus?.qwen_loading || graderStatus?.warmup_running)
                      ? <span style={{ width:10, height:10, borderRadius:'var(--r-round)', border:'2px solid currentColor', borderTopColor:'transparent', animation:'spin .8s linear infinite' }}/>
                      : undefined}>
                    {graderStatus?.qwen_loading ? 'Loading Engine…' : graderStatus?.warmup_running ? 'Calibrating…' : 'Start Culling'}
                  </Button>
                </div>
              );
            })()}
          </div>
        </div>
      )}

      {/* Export modal */}
      {exportModal && (
        <ExportModal
          photos={filterGrade ? filteredPhotos : photos}
          filterGrade={filterGrade}
          onClose={() => setExportModal(false)}
        />
      )}

      {/* ── Header ─────────────────────────────────────────────── */}
      {/* overflow-x-auto keeps the toolbar's minimum width from stretching the
          whole app: on a narrow window the bar scrolls internally instead of
          pushing every full-width section (sheet, count, panels) past the
          viewport edge — the bug that clipped Grade and the photo count. */}
      <header className="shrink-0 px-3 pt-2">
        {/* Detached chrome: a floating glass pill inset to the grid gutter,
            elevated one step above the sheet rather than a surface strip
            bolted to the window edge. min-w-0 lets the row scroll instead of
            stretching its parent (see note above). */}
        {/* Wraps rather than scrolls. Measured on a populated library, this row
            wants 1917px of content; it therefore overflowed at EVERY window
            width, 1920 included. With overflow-x-auto and a hidden scrollbar
            that did not read as "scroll me" — it read as the view tabs and
            Re-grade not existing, because the only affordance was a horizontal
            drag on an invisible bar. Wrapping spends height, and only when the
            window is actually too narrow, to keep every control reachable.
            min-h rather than h so a single-row header is unchanged. */}
        <div className="glass elev-2 flex min-h-toolbar min-w-0 flex-wrap items-center gap-y-1 gap-x-2 rounded-md border border-line-strong px-2 py-1">

        {/* Brand — aperture mark in the grease-pencil colour. The one warm
            pixel in the chrome: it is the product's signature, the same
            reservation a physical camera brand earns on its dial. */}
        <div className="flex shrink-0 items-center gap-1 pr-1" title={`FrameGrade v${APP_VERSION}`}>
          <Aperture size={15} strokeWidth={1.8} style={{ color: T.mark }}/>
          <span className="text-md text-ink"
                style={{ fontFamily: 'var(--font-display)', fontWeight: 650, letterSpacing: 'var(--track-brand)' }}>FrameGrade</span>
        </div>
        <div className="h-4 w-px shrink-0 bg-line-strong"/>

        <Button onClick={openBrowser} title="Open folder" icon={<FolderOpen size={13}/>}>
          {photos.length > 0 ? (folders.length > 1 ? `${folders.length} folders` : folder.split(/[\\/]/).pop()) : 'Open folder'}
        </Button>
        {photos.length > 0 && (
          <Button onClick={openAddFolder} title="Add another folder" variant="quiet"
            icon={<span className="text-md leading-none">+</span>}>
            Add folder
          </Button>
        )}

        <div className="flex-1"/>

        {/* Preset — hidden; value retained for grading logic */}

        {/* Model download. Progress is carried by a luminance fill behind the
            label rather than a coloured badge — it is informational, not an
            alarm, so it stays in the neutral register. */}
        {graderStatus?.qwen_download_pct != null && (
          <span className="relative inline-flex h-6 shrink-0 items-center gap-1 overflow-hidden rounded-sm border border-line-strong bg-surface px-2 text-ink-2">
            <span
              className="absolute inset-y-0 left-0 bg-raised-hover transition-[width] duration-slow ease"
              style={{ width: `${graderStatus.qwen_download_pct}%` }}
            />
            <span className="t-label relative !text-ink-3">Model</span>
            <span className="t-num relative text-xs">{graderStatus.qwen_download_pct}%</span>
          </span>
        )}

        {/* Grader mode indicator */}
        {graderStatus && (() => {
          const m = graderStatus.last_mode;
          const isIqaHeads = m === 'iqa_heads';
          const isClip     = m === 'clip_only';
          const isIdle     = m === 'idle' || !m;
          const dot   = isIqaHeads ? T.gradeStrong : isClip ? T.alarmWarn : T.ink3;
          const label = isIqaHeads ? 'Deep Edit' : isClip ? 'Scout Mode' : 'Ready';
          const tip   = graderStatus.last_error ? `Error: ${graderStatus.last_error}` :
                        isIqaHeads ? 'Full vision pipeline — composition, light, and moment scored' :
                        isClip     ? 'Fast contact-sheet pass — style matching only' :
                        'No grading run yet';
          if (isIdle) return null;
          void dot; // grader mode is informational, so it stays neutral
          return <Chip label={label} title={tip} />;
        })()}

        {/* Cold judge — informational only. A cold Qwen judge adds a one-time
            warm-up to the first grade; without this chip that latency reads as
            a hang. Neutral tone: nothing is wrong, something is pending. */}
        {graderStatus && graderStatus.qwen_warm === false && (
          <Chip label="Judge" title="Vision judge is cold — the first grade includes a one-time warm-up. Later batches run at full speed." />
        )}

        {/* GPU / CPU compute chip */}
        {graderStatus && (() => {
          const dev = graderStatus.compute_device;
          if (!dev) return null;
          const isGpu  = dev === 'gpu';
          const free   = graderStatus.vram_free_gb;
          const total  = graderStatus.vram_total_gb;
          const gpuName = graderStatus.gpu_name;
          const vramStr = free != null && total != null
            ? `${free.toFixed(1)} / ${total.toFixed(1)} GB VRAM`
            : free != null ? `${free.toFixed(1)} GB free` : '';
          const tip = isGpu
            ? [gpuName, vramStr].filter(Boolean).join(' · ')
            : 'Models running on CPU — no CUDA GPU detected or VRAM too low';
          // GPU is the expected state, so it is silent — and silence includes
          // the number. Free VRAM does not change what the photographer does
          // next (the grade floor the app actually enforces is system RAM), so
          // in the expected case the chip just confirms which device is in use
          // and leaves the figure to the tooltip. Falling back to CPU is the
          // case worth flagging: it means a run will be far slower.
          return (
            <Chip
              label={isGpu ? 'GPU' : 'CPU'}
              title={tip}
              tone={isGpu ? 'neutral' : 'warn'}
            />
          );
        })()}

        {/* System RAM chip — live (polled every 2 s), tells the user whether it's clear to grade.
            Non-neutral states are clickable: the popover says what to DO, not just that
            something is wrong. Silence (clear) needs no popover. */}
        {(sysRam || graderStatus) && (() => {
          const r = ramReadiness(sysRam ?? graderStatus);
          if (r.level === 'unknown') return null;
          // The one chip that has genuinely earned its colour on this machine:
          // two culls died tonight when free memory fell under the encoder's
          // load floor. Clear stays neutral so tight and critical actually read.
          const tone = ({ clear: 'neutral', tight: 'warn', critical: 'crit' } as const)[r.level];
          // Served by /api/system/ram as ram_min_gb (_GRADE_MIN_RAM_GB), read
          // ONCE by ramReadiness and handed back as r.min. Never restate it as
          // a literal here: three copies of the floor were baked into display
          // strings and every one went stale the moment the gate moved from
          // 1.8 to 3.8, each of them under-warning the photographer.
          const ramFloorGb = r.min;
          const guidance = r.level === 'critical'
            ? `Browsing the library is safe. Scan and Re-grade will be refused until ~${ramFloorGb} GB is free — close browser tabs or other apps, then retry.`
            : r.level === 'tight'
            ? 'Grading will run, but a long cull may drop to Scout Mode (CLIP-only scoring). Closing heavy apps first keeps the full vision pipeline alive.'
            : 'Clear to grade — the full vision pipeline will run.';
          return (
            <span className="relative inline-flex shrink-0">
              <button
                onClick={() => r.level !== 'clear' && setRamPopoverOpen(v => !v)}
                className={cn('border-0 bg-transparent p-0', r.level !== 'clear' && 'cursor-pointer')}
                title={r.level !== 'clear' ? 'What should I do?' : undefined}>
                <Chip label="RAM" title={r.tip} tone={tone} numeric value={r.readout} />
              </button>
              {ramPopoverOpen && r.level !== 'clear' && (
                <div className="glass absolute left-0 top-8 z-50 w-panel rounded-md border border-line-strong p-3 elev-2 animate-fade-in">
                  <p className="t-label mb-2">Memory</p>
                  <p className="t-num mb-2 text-sm text-ink">
                    {r.free?.toFixed(1)} GB free{r.total != null ? <> of <span className="t-num">{r.total?.toFixed(1)}</span> GB</> : null}
                    {r.percent != null && <> · <span className="t-num">{r.percent.toFixed(0)}%</span> in use</>}
                  </p>
                  <p className="mb-2 text-xs leading-prose text-ink-2">{guidance}</p>
                  <p className="text-xs text-ink-4">Polled live every 2 s. The grade floor is <span className="t-num">{ramFloorGb} GB</span> free.</p>
                </div>
              )}
            </span>
          );
        })()}

        {/* Grade filters. Mid carries a neutral dot rather than its own hue,
            matching the grid: Strong is marked, Weak is dimmed, Mid is silence. */}
        {isDone && (
          <Segmented
            className="animate-fade-in"
            value={filterGrade as 'Strong' | 'Mid' | 'Weak' | null}
            onChange={(v) => setFilterGrade(filterGrade === v ? null : v)}
            options={[
              { value: 'Strong', label: 'Strong', count: picks,   dot: T.gradeStrong },
              { value: 'Mid',    label: 'Mid',    count: mids,    dot: T.ink4 },
              { value: 'Weak',   label: 'Weak',   count: rejects, dot: T.gradeWeak },
            ]}
          />
        )}

        {isDone && graderUsed && (
          <Chip
            label={graderUsed === 'deep' ? 'Deep' : graderUsed === 'scan' ? 'Scan' : 'Fast'}
            title={graderUsed === 'deep'
              ? 'Deep grade: each photo was read in detail (highest accuracy).'
              : graderUsed === 'scan'
              ? 'Scan pass: quick look only, technical scoring skipped (fastest).'
              : 'Fast grade: standard quality scoring. Turn on Deep Grade (and free memory) for the detailed read.'}
          />
        )}

        {isDone && <div className="h-4 w-px shrink-0 bg-line-strong"/>}

        {isDone && (
          <Button
            variant={sortScore ? 'solid' : 'quiet'}
            onClick={() => setSortScore(s => s === null ? 'desc' : s === 'desc' ? 'asc' : null)}
            title={sortScore === 'desc' ? 'Sorted: Strong to Weak' : sortScore === 'asc' ? 'Sorted: Weak to Strong' : 'Sort by score'}
            icon={sortScore === 'desc' ? <ArrowDown size={11}/> : sortScore === 'asc' ? <ArrowUp size={11}/> : <ArrowUpDown size={11}/>}
          >
            Score
          </Button>
        )}

        {isDone && <div className="h-4 w-px shrink-0 bg-line-strong"/>}

        {/* Main views. Counts sit in the segment rather than inside the label
            string, so they render in tabular figures and the tab stops resizing
            as duplicates are resolved. */}
        {isDone && (() => {
          /* One number per concept: the tab and the duplicates header both
             read dupStats.totalDups, so they can never disagree. */
          const dupCount = dupStats.totalDups;
          const madeCount = creativeResults.filter((r:any)=>r.success).length;
          return (
            <Segmented
              className="animate-fade-in"
              value={mainTab}
              onChange={(id) => { setMainTab(id); if (id === 'gallery') setLoupeMode('loupe'); }}
              options={[
                { value: 'gallery',    label: 'Gallery',    icon: <LayoutGrid size={11}/> },
                { value: 'duplicates', label: 'Duplicates', icon: <ImageOff size={11}/>,
                  count: dupCount > 0 ? dupCount : undefined },
                { value: 'creative',   label: 'Creative',   icon: <Wand2 size={11}/>,
                  count: madeCount > 0 ? madeCount : undefined },
              ]}
            />
          );
        })()}

        {isDone && mainTab === 'gallery' && (
          <Segmented
            iconOnly
            value={loupeMode}
            onChange={setLoupeMode}
            options={[
              { value: 'loupe', icon: <RectangleHorizontal size={12}/>, title: 'Loupe (E)' },
              { value: 'grid',  icon: <LayoutGrid size={12}/>,          title: 'Grid (G)' },
            ]}
          />
        )}

        {isDone && (
          <Button onClick={() => setExportModal(true)} icon={<Download size={11}/>}>
            Export
          </Button>
        )}

        {isDone && (
          <Button
            icon={<ArrowUpDown size={11}/>}
            title="Move files on disk into Strong / Mid / Weak folders"
            onClick={async () => {
              try {
                const res = await axios.post(`${API}/api/manage/sort-files`, {
                  folder_path: folders[0] || folder,
                  gallery: photos,
                  copy: false,
                });
                notify(`Sorted ${res.data.moved} files into Strong / Mid / Weak`, 'success');
              } catch (err: any) {
                notify(`Could not sort the files. ${err?.response?.data?.detail ?? err.message}`, 'error');
              }
            }}
          >
            Sort files
          </Button>
        )}

        {/* Scan mode. Active state is a luminance step, not a hue — the warm
            colour belongs to the photographer's marks. */}
        {!isGrading && (
          <Button
            variant={scanMode ? 'solid' : 'quiet'}
            onClick={() => setScanMode(v => !v)}
            title={scanMode
              ? 'Scan pass: a quick look at every frame, technical scoring skipped. Click for the full grade.'
              : 'Full grade: quality scoring on every frame. Click for the faster scan pass.'}
            icon={<Zap size={11} fill={scanMode ? 'currentColor' : 'none'}/>}
          >
            Scan
          </Button>
        )}

        {/* Grade. The primary action, but still not accent-coloured: emphasis
            comes from the filled surface and its position at the end of the bar.
            The old pulse animation is gone — a control that throbs at you while
            you are trying to look at photographs is noise. */}
        {isGrading ? (
          <div className="flex h-6 min-w-[180px] shrink-0 items-center gap-2 rounded-sm border border-line-strong bg-raised px-2">
            <div className="h-px flex-1 overflow-hidden rounded-sm bg-well" style={{ height: 3 }}>
              <div
                className="h-full bg-ink-3 transition-[width] duration-slow ease"
                style={{ width: `${Math.max(2, gradeProgress * 100)}%` }}
              />
            </div>
            <span className="t-num shrink-0 text-xs text-ink">
              {Math.round(gradeProgress * 100)}%
            </span>
            {gradeEtaSecs !== null && gradeEtaSecs > 3 && (
              <span className="t-num shrink-0 text-xs text-ink-3">
                {gradeEtaSecs >= 60 ? `${Math.floor(gradeEtaSecs / 60)}m ${gradeEtaSecs % 60}s` : `${gradeEtaSecs}s`}
              </span>
            )}
          </div>
        ) : (
          <Button
            variant="solid"
            onClick={() => handleGrade(true, false)}
            title="Grade every image, replacing existing scores"
            icon={scanMode ? <Zap size={12} fill="currentColor"/> : <Sparkles size={12}/>}
          >
            {isDone ? (scanMode ? 'Re-scan' : 'Re-grade') : (scanMode ? 'Scan' : 'Grade')}
          </Button>
        )}
        </div>
      </header>

      {/* Progress. A plain ink bar — no gradient. A two-stop gradient sweeping
          across a progress bar is decoration that says nothing the width isn't
          already saying, and it sits directly above the photographs. */}
      <div className="shrink-0 px-3"
        role={isGrading ? "progressbar" : undefined}
        aria-label={isGrading ? "Grading progress" : undefined}
        aria-valuenow={isGrading ? Math.round(gradeProgress * 100) : undefined}
        aria-valuemin={isGrading ? 0 : undefined}
        aria-valuemax={isGrading ? 100 : undefined}
        aria-valuetext={isGrading && gradeDesc ? gradeDesc : undefined}>
        <div className="relative overflow-hidden rounded-full bg-line" style={{ height: 'var(--rule)' }}>
          {listLoading && (
            <div className="absolute top-0 h-full animate-sweep bg-ink-3"/>
          )}
          {!listLoading && isGrading && (
            <div className="h-full bg-ink-2 transition-[width] duration-slow ease"
                 style={{ width: `${Math.max(4, gradeProgress * 100)}%` }}/>
          )}
          {!listLoading && !isGrading && isDone && (
            <div className="h-full w-full bg-grade-strong"/>
          )}
        </div>
        {isGrading && gradeDesc && (() => {
          // Surface the live photo counter (e.g. "44/100") that the backend
          // sends in gradeDesc — toSlogan() rewrites the message into a slogan
          // and would otherwise drop it. Keyed by slogan so the slogan fades in
          // on change while the counter updates in place per photo.
          const _count = (gradeDesc.match(/\d+\s*\/\s*\d+/) || [])[0] || '';
          return (
            <div key={toSlogan(gradeDesc)} style={{ padding:'3px 14px 4px', fontSize:'var(--text-xs)', color:T.ink3, fontStyle:'italic', borderBottom:`1px solid ${T.line}`, animation:'fadeIn .4s cubic-bezier(.2,0,0,1)', display:'flex', gap:8, alignItems:'baseline', justifyContent:'space-between' }}>
              <span>{toSlogan(gradeDesc)}</span>
              <span style={{ display:'flex', gap:8, alignItems:'baseline', flexShrink:0 }}>
                {/* Quality tier actually used for this run. The app picks the
                    best encoder that fits available memory, so this can differ
                    run to run — showing it explains why speed/results vary. */}
                {gradeQuality && (
                  <span title={`Analysis quality for this run: ${gradeQuality}`}
                        style={{ fontStyle:'normal', fontSize:'var(--text-xs)', letterSpacing:'var(--track-label)', textTransform:'uppercase',
                                 color:T.ink2, border:`1px solid ${T.line}`, borderRadius:'var(--r-sm)',
                                 padding:'1px 5px', fontWeight:600 }}>
                    {gradeQuality}
                  </span>
                )}
                {_count && <span style={{ fontStyle:'normal', fontVariantNumeric:'tabular-nums', color:T.ink2 }}>{_count}</span>}
              </span>
            </div>
          );
        })()}
        {/* Warm-up transparency — ambient status while background work initialises
            (thumbnail prewarm, model load, niche detect). Hidden during grading,
            which has its own slogan above. */}
        {!isGrading && (() => {
          const warmMsg =
            listLoading ? 'Generating fast-scroll thumbnails…' :
            (graderStatus?.qwen_loading || graderStatus?.warmup_running) ? 'Waking up models…' :
            null;
          if (!warmMsg) return null;
          return (
            <div style={{ padding:'3px 14px 4px', fontSize:'var(--text-xs)', color:T.ink3, borderBottom:`1px solid ${T.line}`, display:'flex', gap:7, alignItems:'center', animation:'fadeIn .4s cubic-bezier(.2,0,0,1)' }}>
              <div style={{ width:9, height:9, borderRadius:'var(--r-round)', border:'2px solid currentColor', borderTopColor:'transparent', animation:'spin .8s linear infinite' }}/>
              <span>{warmMsg}</span>
            </div>
          );
        })()}
      </div>

      {/* ── Rating filter ──────────────────────────────────────────
       * Star ratings are the photographer's own judgement, so this is one of
       * the few bars allowed to show the warm mark colour. */}
      {mainTab === 'gallery' && isDone && (
        <div className="flex h-8 shrink-0 items-center gap-2 px-3">
          <span className="t-label shrink-0">Rating</span>
          <div className="flex gap-1">
            {[1,2,3,4,5].map(n => {
              const active = filterStars === n;
              return (
                <button key={n} onClick={() => setFilterStars(active ? null : n)}
                  title={`${n} star${n > 1 ? 's' : ''}`}
                  className={cn(
                    'flex cursor-pointer items-center gap-1 rounded-sm border px-2 py-px',
                    'transition-colors duration-fast ease',
                    active
                      ? 'border-mark bg-raised'
                      : 'border-line-strong bg-transparent hover:bg-raised',
                  )}>
                  <span className="flex gap-px">
                    {[1,2,3,4,5].map(s => (
                      <svg key={s} width="8" height="8" viewBox="0 0 24 24" strokeWidth="2"
                        fill={s <= n ? T.mark : 'none'}
                        stroke={s <= n ? T.mark : T.ink4}>
                        <polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/>
                      </svg>
                    ))}
                  </span>
                  <span className={cn('t-num min-w-2 text-center text-xs',
                                      active ? 'text-mark-ink' : 'text-ink-3')}>
                    {starCounts[n]}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="h-4 w-px shrink-0 bg-line-strong"/>

          <Button size="sm" variant={filterStars === 0 ? 'solid' : 'quiet'}
            onClick={() => setFilterStars(filterStars === 0 ? null : 0)}>
            Unrated <span className="t-num ml-1 opacity-70">{starCounts[0]}</span>
          </Button>

          {filterStars !== null && (
            <Button size="sm" variant="quiet" onClick={() => setFilterStars(null)}>Clear</Button>
          )}

        </div>
      )}

      {/* ── Body ───────────────────────────────────────────────── */}
      {mainTab === 'gallery' ? (
        <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', minHeight:0 }}>

          {/* Middle row: grid view OR loupe (preview + right panel) */}
          <div style={{ flex:1, display:'flex', minHeight:0, overflow:'hidden' }}>

            {loupeMode === 'grid' && photos.length > 0 && (
              <ErrorBoundary variant="inline" label="Contact sheet">
              <GridView
                photos={filteredPhotos}
                selId={selId}
                onSelect={id => { setSelId(id); if (isDone) setLoupeMode('loupe'); }}
                usedPaths={allUsedPaths}
                selectMode={selectMode}
                setSelectMode={setSelectMode}
                selectedIds={selectedIds}
                setSelectedIds={setSelectedIds}
                onCreateSequence={handleCreateFromSelection}
                onAutoSequence={handleGenerate}
                nicheDetecting={nicheDetecting}
                dupesCount={redacted.size}
                showDuplicates={showDuplicates}
                onToggleDupes={() => setShowDuplicates(v => !v)}
                shownCount={filteredPhotos.length}
              />
              </ErrorBoundary>
            )}

            {(loupeMode === 'loupe' || photos.length === 0) && (<>

            {/* Center preview */}
            {/* The stage. `bg-well` is the one place near-black is correct —
                it sits directly behind a photograph, where a lighter surround
                would wash out the image being judged. */}
            <div className="relative flex min-h-0 min-w-0 flex-1 items-center justify-center overflow-hidden bg-well">
              {photos.length === 0 ? (
                <WelcomeStage catalogBanner={catalogBanner} onOpenFolder={openBrowser}
                  onResume={handleResume}
                  onStartFresh={() => { axios.post(`${API}/api/catalog/clear`); setCatalogBanner(false); }}/>
              ) : sel ? (
                <ErrorBoundary variant="inline" label="Loupe">
                <LoupeStage
                  sel={sel} selId={selId} setSelId={setSelId}
                  loupePreviewFailed={loupePreviewFailed} setLoupePreviewFailed={setLoupePreviewFailed}
                  loupeRetry={loupeRetry} setLoupeRetry={setLoupeRetry}
                  photoNatDims={photoNatDims} setPhotoNatDims={setPhotoNatDims}
                  selectedIds={selectedIds} setSelectedIds={setSelectedIds}
                  showEyeOverlay={showEyeOverlay} setShowEyeOverlay={setShowEyeOverlay}
                  showHeatmap={showHeatmap} critTrigger={critTrigger} heatmapB64={heatmapB64} heatmapLoading={heatmapLoading} toggleHeatmap={toggleHeatmap}
                  isAuditModeActive={isAuditModeActive} isGraded={isGraded}
                  hasPrev={hasPrev} hasNext={hasNext} selIdx={selIdx} filteredPhotos={filteredPhotos}
                  handleCreateFromSelection={handleCreateFromSelection} handleGenerate={handleGenerate}
                />
                </ErrorBoundary>
              ) : null}
            </div>

            {/* Resize handle */}
            {photos.length > 0 && (
            <div
              onMouseDown={onResizeDown}
              style={{ width:3, cursor:'col-resize', flexShrink:0, background:'transparent', transition:'background .25s ease' }}
              onMouseEnter={e => (e.currentTarget.style.background = T.raisedHover)}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            />
            )}

            {/* Right panel */}
            {photos.length > 0 && <AnalysisPanel
              sel={sel} selId={selId} setSelId={setSelId} photos={photos} rightW={rightW}
              isGraded={isGraded} isDone={isDone}
              isAuditModeActive={isAuditModeActive} setIsAuditModeActive={setIsAuditModeActive}
              critTrigger={critTrigger} setCritTrigger={setCritTrigger}
              juryCritique={juryCritique} juryLoading={juryLoading} handleJuryCritique={handleJuryCritique}
              parseCritique={parseCritique} deepCritique={deepCritique} setDeepCritique={setDeepCritique}
              deepCritiqueLoading={deepCritiqueLoading} setDeepCritiqueLoading={setDeepCritiqueLoading}
              reasoningOverlayUrl={reasoningOverlayUrl} buildReasoningFromBreakdown={buildReasoningFromBreakdown}
              infoTab={infoTab} setInfoTab={setInfoTab}
              selectedIds={selectedIds} setSelectedIds={setSelectedIds}
              handleCopyPath={handleCopyPath} handleSetStars={handleSetStars} setMainTab={setMainTab} copied={copied}
              handleGenerate={handleGenerate} handleCreateFromSelection={handleCreateFromSelection}
              hasPrev={hasPrev} hasNext={hasNext} selIdx={selIdx} filteredPhotos={filteredPhotos}
            />}

            </>)}
          </div>

          {/* ── Filmstrip (loupe mode only) ─────────────────────── */}
          {loupeMode === 'loupe' && photos.length > 0 && (
          <div style={{ flexShrink:0, background:T.surface, borderTop:`1px solid ${T.line}`, display:'flex', flexDirection:'column' }}>
            <div style={{ height:20, flexShrink:0, display:'flex', alignItems:'center', justifyContent:'space-between', padding:'0 12px', borderBottom:`1px solid ${T.line}` }}>
              <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                <span style={{ fontSize:'var(--text-xs)', color:T.ink3, fontWeight:600, letterSpacing:'var(--track-label)', textTransform:'uppercase' }}>Library</span>
                {/* Tweaks toggle */}
                <button title="Filmstrip settings" onClick={() => setShowTweaks(v => !v)}
                  style={{ display:'flex', alignItems:'center', justifyContent:'center', width:18, height:16, cursor:'pointer', background:showTweaks ? T.raisedHover : 'transparent', color:showTweaks ? T.ink : T.ink3, border:'none', borderRadius:'var(--r-sm)', transition:'all .25s cubic-bezier(.2,0,0,1)' }}>
                  <SlidersHorizontal size={9}/>
                </button>
              </div>
              <span style={{ fontSize:'var(--text-xs)', color:T.ink3, fontVariantNumeric:'tabular-nums', display:'flex', alignItems:'center', gap:5 }}>
                {isGrading && <span style={{ display:'inline-block', width:5, height:5, border:`1.5px solid ${T.ink3}`, borderTopColor:'transparent', borderRadius:'var(--r-round)', animation:'spin .8s linear infinite' }}/>}
                {isDone
                  ? <><span style={{ color:T.gradeStrong }}>{picks} picks</span>{'  ·  '}<span style={{ color:T.gradeWeak }}>{rejects} rejects</span>{'  ·  '}{photos.length} total</>
                  : `${photos.length} photos`}
              </span>
            </div>
            {/* Tweaks panel */}
            {showTweaks && (
              <div style={{ flexShrink:0, display:'flex', alignItems:'center', gap:16, padding:'6px 12px', borderBottom:`1px solid ${T.line}`, background:T.raised }}>
                <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                  <span style={{ fontSize:'var(--text-xs)', color:T.ink3, whiteSpace:'nowrap' }}>Thumb size</span>
                  <input type="range" min={60} max={130} step={4} value={filmThumbH}
                    onChange={e => setFilmThumbH(Number(e.target.value))}
                    style={{ width:80, accentColor:T.ink, cursor:'pointer' }}/>
                  <span style={{ fontSize:'var(--text-xs)', color:T.ink2, fontVariantNumeric:'tabular-nums', minWidth:22 }}>{filmThumbH}</span>
                </div>
                <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                  <span style={{ fontSize:'var(--text-xs)', color:T.ink3 }}>Filenames</span>
                  <button onClick={() => setShowFilename(v => !v)}
                    style={{ position:'relative', width:28, height:16, borderRadius:'var(--r-md)', border:'none', cursor:'pointer', padding:0, background:showFilename ? T.ink : T.lineStrong, transition:'background .25s ease' }}>
                    <span style={{ position:'absolute', top:2, left:showFilename ? 13 : 2, width:12, height:12, borderRadius:'var(--r-round)', background:T.ink, transition:'left .22s cubic-bezier(.2,0,0,1)', boxShadow:`0 1px 2px ${T.well}` }}/>
                  </button>
                </div>
              </div>
            )}
            <div style={{ height: filmThumbH + (showFilename ? 18 : 0) + 12, flexShrink: 0 }}>
              <Filmstrip photos={filteredPhotos} selId={selId} onSelect={setSelId}
                usedPaths={allUsedPaths} selectedIds={selectedIds}
                h={filmThumbH} showFn={showFilename}/>
            </div>
          </div>
          )}
        </div>

      ) : mainTab === 'duplicates' ? (
        /* ── Duplicates grid view — see components/views/SimilarShots.tsx ── */
        <ErrorBoundary variant="inline" label="Duplicates">
        <SimilarShots
          groups={dupStats.groups}
          totalDups={dupStats.totalDups}
          shownCount={dupGroupsShown}
          onRevealMore={revealMoreDupGroups}
          onOpenPhoto={(id) => { setMainTab('gallery'); setSelId(id); setLoupeMode('loupe'); }}
          onExport={() => setExportModal(true)}
        />
        </ErrorBoundary>

      ) : mainTab === 'creative' ? (
        /* ── Creative Direction view ───────────────────────────── */
        <ErrorBoundary variant="inline" label="Creative Director">
        <CreativeDirector
          photos={photos} creativeResults={creativeResults} creativeLoading={creativeLoading}
          engineHealth={engineHealth}
          creativePrompt={creativePrompt} setCreativePrompt={setCreativePrompt}
          ragPdfs={ragPdfs} ragUploading={ragUploading} handleRagUpload={handleRagUpload} handleRagClear={handleRagClear}
          creativeAnchor={creativeAnchor} setCreativeAnchor={setCreativeAnchor}
          seqMode={seqMode} setSeqMode={setSeqMode}
          handleRunCreativeDirection={handleRunCreativeDirection} handleSaveSequence={handleSaveSequence}
          sequenceSaving={sequenceSaving}
          creativeSelection={creativeSelection} creativeFallback={creativeFallback}
          creativeProgress={creativeProgress} creativeStage={creativeStage}
          creativeCount={creativeCount} setCreativeCount={setCreativeCount}
          usedCount={usedCount} handleClearUsed={handleClearUsed}
          pegFile={pegFile} setPegFile={setPegFile} pegHash={pegHash} setPegHash={setPegHash}
          pegLoading={pegLoading} handlePegUpload={handlePegUpload}
        />
        </ErrorBoundary>
      ) : null}

      {/* ── Status bar ─────────────────────────────────────────── */}
      <div className="flex h-6 shrink-0 items-center gap-4 border-t border-line bg-surface px-3">
        <span className="t-num flex-1 truncate text-xs text-ink-2">
          {sel ? sel.path.split(/[\\/]/).pop() : 'Open a folder to begin'}
        </span>
        <div className="flex shrink-0 gap-3">
          {[['⌘K','Commands'],['← →','Navigate'],['1–5','Rate'],['0','Clear'],['X','Used'],['G','Grid'],['E','Loupe']].map(([k, a]) => (
            <KbdHint key={k} keys={k} label={a}/>
          ))}
        </div>
      </div>

      {/* ── Folder browser modal ────────────────────────────────── */}
      {/* ── Command palette (⌘K / Ctrl-K) — the keyboard's complete control
              surface. Self-registering listener; App only supplies actions. ── */}
      <CommandPalette actions={[
        { id: 'grade',    label: 'Grade folder',     group: 'Grade', hint: 'run the grader',                 run: () => { void handleGrade(); } },
        { id: 'loupe',    label: 'Open loupe',       group: 'View',  kbd: 'E', hint: 'view current',         run: () => setLoupeMode('loupe') },
        { id: 'grid',     label: 'Back to grid',     group: 'View',  kbd: 'G', hint: 'contact sheet',        run: () => setLoupeMode('grid') },
        { id: 'gallery',  label: 'Gallery view',     group: 'View',                                          run: () => setMainTab('gallery') },
        { id: 'dupes',    label: 'Duplicates view',  group: 'View',  hint: 'similar shots',                  run: () => setMainTab('duplicates') },
        { id: 'creative', label: 'Creative Director', group: 'View', hint: 'sequence builder',               run: () => setMainTab('creative') },
        { id: 'open',     label: 'Open folder…',     group: 'Library', hint: 'browse',                       run: () => openBrowser() },
        { id: 'add',      label: 'Add folder…',      group: 'Library',                                       run: () => openAddFolder() },
        { id: 'clear',    label: 'Clear used marks', group: 'Library',                                       run: () => handleClearUsed() },
        { id: 'export',   label: 'Export sequence…', group: 'Export', hint: `${carousel.length} in sequence`, run: () => setExportModal(true) },
      ]}/>

      {showBrowser && (
        <FolderBrowser
          mode={browserMode}
          bPath={bPath}
          setBPath={setBPath}
          bFolders={bFolders}
          bImages={bImages}
          bSelFolders={bSelFolders}
          setBSelFolders={setBSelFolders}
          loading={bLoading}
          onNavigate={loadBrowser}
          onGoUp={goUp}
          onFolderClick={(e, p, i) => handleBrowserFolderClick(e, p, i)}
          onAddFolders={async (fs) => { for (const nf of fs) await handleAddFolder(nf); }}
          onUseFolder={() => { setFolder(bPath); setPhotos([]); setSelId(null); setFolders([]); }}
          onClose={() => setShowBrowser(false)}
        />
      )}
    </div>
  );
}



