import React, { useEffect } from "react";
import { X, CheckCircle2, XCircle, AlertTriangle, AlertCircle, Calendar, Hash, Image as ImageIcon } from "lucide-react";
import type { InspectionResult } from "../types/inspection";
import { HeatmapViewer } from "./HeatmapViewer";
import { ScoreGauge } from "./ScoreGauge";

interface InspectionDetailModalProps {
  inspection: InspectionResult | null;
  onClose: () => void;
}

export const InspectionDetailModal: React.FC<InspectionDetailModalProps> = ({
  inspection,
  onClose,
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  if (!inspection) return null;

  const resultType = inspection.result.toLowerCase();
  const inspectionId = inspection.inspection_id ?? inspection.id ?? 0;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontWeight: 700,
                fontSize: "1.1rem",
                color: "var(--text-main)",
              }}
            >
              Inspection Record #{inspectionId}
            </span>
            <span className={`badge-decision ${resultType}`}>
              {inspection.result}
            </span>
          </div>

          <button type="button" className="icon-btn" onClick={onClose} title="Close Modal">
            <X size={18} />
          </button>
        </div>

        {/* Modal Body */}
        <div className="modal-body">
          {/* Metadata Bar */}
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "1.25rem",
              fontSize: "0.8rem",
              color: "var(--text-muted)",
              background: "var(--bg-card)",
              padding: "0.75rem 1rem",
              borderRadius: "var(--radius-md)",
              border: "1px solid var(--border-subtle)",
              fontFamily: "var(--font-mono)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
              <Calendar size={14} color="var(--text-dim)" />
              <span>{new Date(inspection.timestamp).toLocaleString()}</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
              <Hash size={14} color="var(--text-dim)" />
              <span>ID: {inspectionId}</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
              <ImageIcon size={14} color="var(--text-dim)" />
              <span>{inspection.filename || "camera-capture.jpg"}</span>
            </div>
          </div>

          {/* Decision Summary Banner */}
          <div className={`decision-hero ${resultType}`}>
            <div className="decision-left">
              <div className="decision-icon-box">
                {inspection.result === "PASS" && <CheckCircle2 size={28} />}
                {inspection.result === "FAIL" && <XCircle size={28} />}
                {inspection.result === "REVIEW" && <AlertTriangle size={28} />}
                {inspection.result === "INSPECTION_INVALID" && <AlertCircle size={28} />}
              </div>
              <div>
                <div className="decision-title">{inspection.result}</div>
                <div className="decision-subtext">{inspection.reason || "Inspection concluded"}</div>
              </div>
            </div>

            <div className="decision-confidence-box">
              <div className="confidence-val">
                {inspection.confidence != null ? `${inspection.confidence.toFixed(0)}%` : "—"}
              </div>
              <div className="confidence-lbl">Model Confidence</div>
            </div>
          </div>

          {/* Anomaly Gauge */}
          <ScoreGauge
            score={inspection.anomaly_score}
            threshold={inspection.threshold}
            confidence={inspection.confidence}
          />

          {/* Quality Gates Checklist */}
          <div className="gates-grid">
            <div className={`gate-card ${inspection.image_quality_valid !== false ? "valid" : "invalid"}`}>
              <div className="gate-status-icon">
                {inspection.image_quality_valid !== false ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
              </div>
              <div className="gate-info">
                <span className="gate-title">Image Quality</span>
                <span className="gate-desc">
                  {inspection.quality_reason || (inspection.image_quality_valid !== false ? "Acceptable" : "Degraded")}
                </span>
              </div>
            </div>

            <div className={`gate-card ${inspection.product_detected !== false ? "valid" : "invalid"}`}>
              <div className="gate-status-icon">
                {inspection.product_detected !== false ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
              </div>
              <div className="gate-info">
                <span className="gate-title">Product Detected</span>
                <span className="gate-desc">
                  {inspection.product_detected !== false ? "Cap in Frame" : "No Product"}
                </span>
              </div>
            </div>

            <div className={`gate-card ${inspection.alignment_valid !== false ? "valid" : "invalid"}`}>
              <div className="gate-status-icon">
                {inspection.alignment_valid !== false ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
              </div>
              <div className="gate-info">
                <span className="gate-title">Alignment</span>
                <span className="gate-desc">
                  {inspection.alignment_valid !== false ? "Centered (80% ROI)" : "Off-Center"}
                </span>
              </div>
            </div>
          </div>

          {/* Image & Heatmap Visualizer */}
          <HeatmapViewer
            imageUrl={inspection.image_url}
            heatmapUrl={inspection.heatmap_url}
            altText={`Inspection #${inspectionId} heatmap`}
          />
        </div>
      </div>
    </div>
  );
};
