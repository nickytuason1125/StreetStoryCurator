import { useEffect } from "react";
import { X } from "lucide-react";

export function Toast({
  message,
  type = "info",
  onClose,
}: {
  message:  string;
  type?:    "success" | "error" | "info";
  onClose:  () => void;
}) {
  useEffect(() => {
    const t = setTimeout(onClose, 3500);
    return () => clearTimeout(t);
  }, [onClose]);

  const colors = {
    success: "bg-green-900/90 text-green-100 border-green-700",
    error:   "bg-red-900/90   text-red-100   border-red-700",
    info:    "bg-blue-900/90  text-blue-100  border-blue-700",
  };

  return (
    <div
      className={`fixed top-4 right-4 px-4 py-3 rounded-lg border shadow-xl z-50 flex items-center gap-3 min-w-[280px] ${colors[type]}`}
    >
      <span className="text-sm font-medium flex-1">{message}</span>
      <button onClick={onClose} className="opacity-70 hover:opacity-100">
        <X size={14} />
      </button>
    </div>
  );
}
