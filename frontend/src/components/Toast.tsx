import { useEffect } from "react";
import { X } from "lucide-react";
import { cn } from "../lib/cn";

/* Notification.
 *
 * Previously styled with Tailwind's stock palette (bg-green-900, border-red-700,
 * rounded-lg). Those class names stopped existing when the theme was replaced by
 * the token scales, so the toast was rendering with no surface at all — text
 * floating over whatever was behind it. Rebuilt on tokens.
 *
 * Only an error takes colour. A success toast already says the thing succeeded;
 * painting it green as well is the interface repeating itself, and green chrome
 * beside a photograph is exactly what the palette is trying to avoid.
 */

export function Toast({
  message,
  type = "info",
  onClose,
}: {
  message: string;
  type?: "success" | "error" | "info";
  onClose: () => void;
}) {
  useEffect(() => {
    // Errors stay put — they usually name something the user has to act on, and
    // 3.5s is not enough to read a path or a reason.
    if (type === "error") return;
    const t = setTimeout(onClose, 4000);
    return () => clearTimeout(t);
  }, [onClose, type]);

  return (
    <div
      role={type === "error" ? "alert" : "status"}
      aria-live={type === "error" ? "assertive" : "polite"}
      className={cn(
        "animate-fade-in fixed right-4 top-4 z-50 flex min-w-[280px] max-w-[420px]",
        "items-start gap-3 rounded-md border bg-surface px-4 py-3 elev-3",
        type === "error" ? "border-alarm-crit" : "border-line-strong",
      )}
    >
      <span className={cn("flex-1 text-sm", type === "error" ? "text-alarm-crit" : "text-ink")}>
        {message}
      </span>
      <button
        onClick={onClose}
        aria-label="Dismiss"
        className="shrink-0 cursor-pointer rounded-sm border-0 bg-transparent p-0 text-ink-3 transition-colors duration-fast ease hover:text-ink"
      >
        <X size={13} aria-hidden="true" />
      </button>
    </div>
  );
}
