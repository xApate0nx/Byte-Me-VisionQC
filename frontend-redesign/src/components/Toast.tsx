import React, { useEffect } from "react";
import { CheckCircle2, AlertTriangle, XCircle, Info, X } from "lucide-react";

export interface ToastMessage {
  id: string;
  type: "success" | "error" | "warning" | "info";
  message: string;
}

interface ToastProps {
  toast: ToastMessage | null;
  onDismiss: () => void;
}

export const Toast: React.FC<ToastProps> = ({ toast, onDismiss }) => {
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => {
      onDismiss();
    }, 4000);
    return () => clearTimeout(timer);
  }, [toast, onDismiss]);

  if (!toast) return null;

  return (
    <div className={`toast-floating ${toast.type}`}>
      {toast.type === "success" && <CheckCircle2 size={18} color="var(--color-pass)" />}
      {toast.type === "error" && <XCircle size={18} color="var(--color-fail)" />}
      {toast.type === "warning" && <AlertTriangle size={18} color="var(--color-review)" />}
      {toast.type === "info" && <Info size={18} color="#38bdf8" />}

      <span style={{ fontWeight: 500 }}>{toast.message}</span>

      <button
        type="button"
        onClick={onDismiss}
        style={{ marginLeft: 8, opacity: 0.7, color: "inherit", padding: 2 }}
        title="Dismiss"
      >
        <X size={14} />
      </button>
    </div>
  );
};
