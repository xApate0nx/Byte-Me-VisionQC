import React, { useRef, useState, useEffect, useCallback } from "react";
import {
  Camera,
  CameraOff,
  Upload,
  Sparkles,
  Zap,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  AlertCircle,
  Layers,
  Flashlight,
  Video,
} from "lucide-react";
import { useCamera } from "../hooks/useCamera";
import { api } from "../services/api";
import type { InspectionResult } from "../types/inspection";
import { RoiGuideOverlay } from "../components/RoiGuideOverlay";
import { ScoreGauge } from "../components/ScoreGauge";
import { HeatmapViewer } from "../components/HeatmapViewer";

interface LiveInspectionPageProps {
  threshold: number;
  onInspectionFinished: () => void;
  soundCuePass: () => void;
  soundCueFail: () => void;
  soundCueReview: () => void;
  showToast: (msg: string, type: "success" | "error" | "warning" | "info") => void;
}

export const LiveInspectionPage: React.FC<LiveInspectionPageProps> = ({
  threshold,
  onInspectionFinished,
  soundCuePass,
  soundCueFail,
  soundCueReview,
  showToast,
}) => {
  const camera = useCamera();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [currentResult, setCurrentResult] = useState<InspectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [inspectionTimeMs, setInspectionTimeMs] = useState<number | null>(null);

  const executeInspection = useCallback(
    async (file: Blob, filename = "capture.jpg") => {
      setLoading(true);
      const start = performance.now();
      try {
        const result = await api.inspect(file, filename);
        const duration = Math.round(performance.now() - start);
        setInspectionTimeMs(duration);
        setCurrentResult(result);
        onInspectionFinished();

        // Sound cues
        if (result.result === "PASS") {
          soundCuePass();
        } else if (result.result === "FAIL") {
          soundCueFail();
        } else if (result.result === "REVIEW") {
          soundCueReview();
        }

        showToast(
          `Inspection completed: ${result.result} (Score: ${result.anomaly_score?.toFixed(3) ?? "N/A"})`,
          result.result === "PASS"
            ? "success"
            : result.result === "FAIL"
            ? "error"
            : "warning"
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Inspection failed.";
        showToast(msg, "error");
      } finally {
        setLoading(false);
      }
    },
    [onInspectionFinished, soundCuePass, soundCueFail, soundCueReview, showToast]
  );

  const handleCaptureAndInspect = async () => {
    try {
      const blob = await camera.capture();
      await executeInspection(blob, "webcam-capture.jpg");
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Failed to capture image.", "error");
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      executeInspection(file, file.name);
      e.target.value = "";
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      executeInspection(file, file.name);
    }
  };

  // 1-Click Sample Test Loaders
  const handleLoadSample = async (type: "normal" | "defect") => {
    try {
      const sampleUrl =
        type === "normal"
          ? "/samples/normal_cap.jpg"
          : "/samples/defective_cap.jpg";
      const sampleName =
        type === "normal" ? "sample-normal-cap.jpg" : "sample-defective-cap.jpg";
      const file = await api.loadSampleImage(sampleUrl, sampleName);
      await executeInspection(file, sampleName);
    } catch (err) {
      showToast("Could not load sample test image.", "error");
    }
  };

  // Keyboard shortcut: Space to capture & inspect if camera active
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && camera.active && !loading) {
        e.preventDefault();
        handleCaptureAndInspect();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [camera.active, loading]);

  const resultType = currentResult?.result.toLowerCase();

  return (
    <div className="inspection-grid">
      {/* LEFT PANEL: Camera Viewport & Controls */}
      <div className="card-panel">
        <div className="panel-header">
          <div className="panel-title">
            <Video size={18} color="#38bdf8" />
            <span>Inspection Viewport</span>
          </div>

          {/* Camera Device Switcher */}
          {camera.devices.length > 1 && (
            <select
              style={{
                background: "var(--bg-muted)",
                color: "var(--text-main)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-sm)",
                padding: "2px 8px",
                fontSize: "0.75rem",
              }}
              value={camera.selectedDeviceId}
              onChange={(e) => camera.switchDevice(e.target.value)}
            >
              {camera.devices.map((d) => (
                <option key={d.deviceId} value={d.deviceId}>
                  {d.label}
                </option>
              ))}
            </select>
          )}
        </div>

        <div className="panel-body">
          {/* Video Viewport / Dropzone */}
          <div
            className={`viewport-container ${dragOver ? "drag-over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            {/* Live Video Feed */}
            <video
              ref={camera.videoRef}
              className="video-feed"
              autoPlay
              playsInline
              muted
              style={{ display: camera.active ? "block" : "none" }}
            />

            {/* Offline Placeholder */}
            {!camera.active && (
              <div className="viewport-placeholder">
                <Camera size={48} strokeWidth={1.5} color="var(--text-dim)" />
                <div>
                  <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-main)" }}>
                    Camera Standby
                  </h3>
                  <p style={{ fontSize: "0.8rem", marginTop: 4 }}>
                    Start the camera or drag and drop a product image here
                  </p>
                </div>
              </div>
            )}

            {/* HUD Status Bar */}
            {camera.active && (
              <div className="viewport-hud">
                <div className="hud-badge">
                  ● LIVE FEED {camera.resolution ? `(${camera.resolution.width}x${camera.resolution.height})` : ""}
                </div>

                {camera.torchAvailable && (
                  <button
                    type="button"
                    className="hud-badge"
                    onClick={camera.toggleTorch}
                    style={{ pointerEvents: "auto", cursor: "pointer" }}
                  >
                    <Flashlight size={12} style={{ display: "inline", marginRight: 4 }} />
                    {camera.torchOn ? "Torch ON" : "Torch OFF"}
                  </button>
                )}
              </div>
            )}

            {/* 80% ROI Reticle Overlay */}
            <RoiGuideOverlay isInspecting={loading} isActive={camera.active} />
          </div>

          {/* Viewport Error Display */}
          {camera.error && (
            <div
              style={{
                padding: "0.65rem 0.95rem",
                borderRadius: "var(--radius-md)",
                background: "rgba(239, 68, 68, 0.1)",
                border: "1px solid rgba(239, 68, 68, 0.3)",
                color: "#fca5a5",
                fontSize: "0.8rem",
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
              }}
            >
              <AlertCircle size={16} />
              <span>{camera.error}</span>
            </div>
          )}

          {/* Primary Action Buttons */}
          <div className="controls-grid">
            <button
              type="button"
              className={camera.active ? "btn-secondary btn-danger" : "btn-secondary"}
              onClick={camera.active ? camera.stop : () => camera.start()}
              disabled={loading}
            >
              {camera.active ? <CameraOff size={16} /> : <Camera size={16} />}
              {camera.active ? "Stop Camera" : "Start Camera"}
            </button>

            <button
              type="button"
              className="btn-primary"
              disabled={!camera.active || loading}
              onClick={handleCaptureAndInspect}
              title="Capture live frame and run PatchCore inference (Spacebar)"
            >
              <Zap size={18} />
              <span>{loading ? "Inspecting..." : "Snap & Inspect"}</span>
              <kbd
                style={{
                  background: "rgba(0,0,0,0.25)",
                  padding: "2px 6px",
                  borderRadius: 4,
                  fontSize: "0.7rem",
                  marginLeft: 4,
                }}
              >
                Space
              </kbd>
            </button>

            <button
              type="button"
              className="btn-secondary"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              title="Upload image file from disk"
            >
              <Upload size={16} />
              Upload
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={handleFileUpload}
            />
          </div>

          {/* Quick Demo Test Samples */}
          <div className="samples-strip">
            <div className="samples-strip-label">
              <Sparkles size={14} color="#f59e0b" />
              <span>Demo Test Samples:</span>
            </div>
            <div className="samples-buttons">
              <button
                type="button"
                className="btn-pill-sample normal"
                onClick={() => handleLoadSample("normal")}
                disabled={loading}
              >
                <CheckCircle2 size={13} />
                Test Normal Cap
              </button>
              <button
                type="button"
                className="btn-pill-sample defect"
                onClick={() => handleLoadSample("defect")}
                disabled={loading}
              >
                <XCircle size={13} />
                Test Defective Cap
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT PANEL: Live Inspection Result & Telemetry */}
      <div className="card-panel">
        <div className="panel-header">
          <div className="panel-title">
            <Layers size={18} color="#10b981" />
            <span>Inspection Intelligence</span>
          </div>

          {inspectionTimeMs !== null && (
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.72rem",
                color: "var(--text-muted)",
              }}
            >
              Latency: {inspectionTimeMs}ms
            </span>
          )}
        </div>

        <div className="panel-body">
          {/* Empty State when no inspection yet */}
          {!currentResult && !loading && (
            <div
              style={{
                padding: "3.5rem 1.5rem",
                textAlign: "center",
                color: "var(--text-dim)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "0.75rem",
              }}
            >
              <div
                style={{
                  width: 56,
                  height: 56,
                  borderRadius: "50%",
                  background: "var(--bg-muted)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "var(--text-muted)",
                }}
              >
                <Layers size={28} />
              </div>
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-main)" }}>
                Awaiting Inspection
              </h3>
              <p style={{ fontSize: "0.85rem", maxWidth: 320 }}>
                Align the water bottle cap within the 80% ROI circle, capture a frame, or select a
                demo sample to inspect.
              </p>
            </div>
          )}

          {/* Shimmer / Scanning State during Inference */}
          {loading && (
            <div
              style={{
                padding: "3rem 1.5rem",
                textAlign: "center",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "1rem",
              }}
            >
              <div
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: "50%",
                  border: "3px solid var(--border-strong)",
                  borderTopColor: "var(--color-primary)",
                  animation: "spin 0.8s linear infinite",
                }}
              />
              <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
              <div>
                <h4 style={{ fontSize: "1rem", fontWeight: 700 }}>PatchCore Anomaly Detection</h4>
                <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 4 }}>
                  Extracting patch features, computing memory bank distance, and generating heatmap...
                </p>
              </div>
            </div>
          )}

          {/* Active Result View */}
          {currentResult && !loading && (
            <>
              {/* Hero Decision Banner */}
              <div className={`decision-hero ${resultType}`}>
                <div className="decision-left">
                  <div className="decision-icon-box">
                    {currentResult.result === "PASS" && <CheckCircle2 size={30} />}
                    {currentResult.result === "FAIL" && <XCircle size={30} />}
                    {currentResult.result === "REVIEW" && <AlertTriangle size={30} />}
                    {currentResult.result === "INSPECTION_INVALID" && <AlertCircle size={30} />}
                  </div>
                  <div>
                    <div className="decision-title">{currentResult.result}</div>
                    <div className="decision-subtext">{currentResult.reason}</div>
                  </div>
                </div>

                <div className="decision-confidence-box">
                  <div className="confidence-val">
                    {currentResult.confidence != null ? `${currentResult.confidence.toFixed(0)}%` : "—"}
                  </div>
                  <div className="confidence-lbl">Decision Confidence</div>
                </div>
              </div>

              {/* Anomaly Score Gauge */}
              <ScoreGauge
                score={currentResult.anomaly_score}
                threshold={currentResult.threshold || threshold}
                confidence={currentResult.confidence}
              />

              {/* Quality Pre-Check Checklist */}
              <div className="gates-grid">
                <div
                  className={`gate-card ${
                    currentResult.image_quality_valid !== false ? "valid" : "invalid"
                  }`}
                >
                  <div className="gate-status-icon">
                    {currentResult.image_quality_valid !== false ? (
                      <CheckCircle2 size={13} />
                    ) : (
                      <XCircle size={13} />
                    )}
                  </div>
                  <div className="gate-info">
                    <span className="gate-title">Quality Gate</span>
                    <span className="gate-desc">
                      {currentResult.quality_reason ||
                        (currentResult.image_quality_valid !== false ? "Pass" : "Blur/Exposure")}
                    </span>
                  </div>
                </div>

                <div
                  className={`gate-card ${
                    currentResult.product_detected !== false ? "valid" : "invalid"
                  }`}
                >
                  <div className="gate-status-icon">
                    {currentResult.product_detected !== false ? (
                      <CheckCircle2 size={13} />
                    ) : (
                      <XCircle size={13} />
                    )}
                  </div>
                  <div className="gate-info">
                    <span className="gate-title">Cap Detected</span>
                    <span className="gate-desc">
                      {currentResult.product_detected !== false ? "Present" : "Missing"}
                    </span>
                  </div>
                </div>

                <div
                  className={`gate-card ${
                    currentResult.alignment_valid !== false ? "valid" : "invalid"
                  }`}
                >
                  <div className="gate-status-icon">
                    {currentResult.alignment_valid !== false ? (
                      <CheckCircle2 size={13} />
                    ) : (
                      <XCircle size={13} />
                    )}
                  </div>
                  <div className="gate-info">
                    <span className="gate-title">80% Alignment</span>
                    <span className="gate-desc">
                      {currentResult.alignment_valid !== false ? "Centered" : "Misaligned"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Visual Heatmap & Comparison */}
              <HeatmapViewer
                imageUrl={currentResult.image_url}
                heatmapUrl={currentResult.heatmap_url}
                altText={`Anomaly heatmap for inspection #${currentResult.inspection_id}`}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
};
