import React, { useState } from "react";
import { Sliders, Save, RotateCcw, Cpu, Database, Info, ShieldCheck, Check } from "lucide-react";
import { api } from "../services/api";

interface SettingsPageProps {
  currentThreshold: number;
  onThresholdUpdated: (newVal: number) => void;
  showToast: (msg: string, type: "success" | "error" | "warning" | "info") => void;
}

const FACTORY_DEFAULT_THRESHOLD = 12.3785223961;

export const SettingsPage: React.FC<SettingsPageProps> = ({
  currentThreshold,
  onThresholdUpdated,
  showToast,
}) => {
  const [val, setVal] = useState<number>(currentThreshold);
  const [saving, setSaving] = useState(false);

  // Sync when currentThreshold changes externally
  React.useEffect(() => {
    setVal(currentThreshold);
  }, [currentThreshold]);

  const handleSave = async () => {
    if (val <= 0) {
      showToast("Threshold must be greater than 0.", "error");
      return;
    }
    setSaving(true);
    try {
      const res = await api.updateThreshold(val);
      onThresholdUpdated(res.threshold);
      showToast(`Threshold updated and persisted: ${res.threshold.toFixed(6)}`, "success");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to update threshold.";
      showToast(msg, "error");
    } finally {
      setSaving(false);
    }
  };

  const applyPreset = (presetVal: number, name: string) => {
    setVal(presetVal);
    showToast(`Loaded preset "${name}" (${presetVal.toFixed(4)}). Click Save to apply.`, "info");
  };

  const adjustBy = (delta: number) => {
    setVal((prev) => Math.max(0.001, Number((prev + delta).toFixed(4))));
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Supervisor Threshold Calibration Card */}
      <div className="card-panel">
        <div className="panel-header">
          <div className="panel-title">
            <Sliders size={18} color="#38bdf8" />
            <span>Supervisor Threshold Calibration</span>
          </div>

          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.75rem",
              background: "var(--bg-muted)",
              padding: "3px 8px",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-subtle)",
              color: "var(--text-muted)",
            }}
          >
            ACTIVE: {currentThreshold.toFixed(6)}
          </span>
        </div>

        <div className="panel-body">
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
            The PatchCore decision threshold defines the boundary separating normal water bottle
            caps from defective anomalies. Scores higher than this threshold (plus the 10% review
            margin) trigger automated rejection.
          </p>

          {/* Stepper & Input controls */}
          <div className="setting-box">
            <div className="setting-row">
              <div>
                <div className="setting-label-title">Decision Threshold Value</div>
                <div className="setting-label-desc">
                  Calibrated anomaly score cutoff point
                </div>
              </div>

              <div className="number-stepper">
                <input
                  type="number"
                  step="0.0001"
                  min="0.0001"
                  max="50"
                  value={val}
                  onChange={(e) => setVal(Number(e.target.value))}
                  className="stepper-input"
                />

                <button
                  type="button"
                  className="btn-primary"
                  onClick={handleSave}
                  disabled={saving}
                  style={{ minWidth: 120 }}
                >
                  {saving ? <RotateCcw size={16} className="spin" /> : <Save size={16} />}
                  <span>{saving ? "Saving..." : "Save"}</span>
                </button>
              </div>
            </div>

            {/* Slider Control */}
            <div className="slider-wrapper">
              <input
                type="range"
                min="5"
                max="25"
                step="0.01"
                value={val}
                onChange={(e) => setVal(Number(e.target.value))}
                className="slider-input"
              />
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "0.72rem",
                  color: "var(--text-dim)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                <span>5.0 (Strict / Sensitive)</span>
                <span>Factory Default: {FACTORY_DEFAULT_THRESHOLD.toFixed(2)}</span>
                <span>25.0 (Permissive / Relaxed)</span>
              </div>
            </div>

            {/* Fine Nudge Steppers */}
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>
                Quick Nudge:
              </span>
              {[
                { label: "-0.50", delta: -0.5 },
                { label: "-0.10", delta: -0.1 },
                { label: "-0.01", delta: -0.01 },
                { label: "+0.01", delta: 0.01 },
                { label: "+0.10", delta: 0.1 },
                { label: "+0.50", delta: 0.5 },
              ].map((item) => (
                <button
                  key={item.label}
                  type="button"
                  className="btn-secondary"
                  style={{ padding: "0.25rem 0.55rem", fontSize: "0.75rem", fontFamily: "var(--font-mono)" }}
                  onClick={() => adjustBy(item.delta)}
                >
                  {item.label}
                </button>
              ))}
            </div>

            {/* Operating Presets */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
                flexWrap: "wrap",
                borderTop: "1px solid var(--border-subtle)",
                paddingTop: "1rem",
              }}
            >
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>
                Presets:
              </span>

              <button
                type="button"
                className="btn-pill-sample"
                onClick={() => applyPreset(10.5, "Strict QC")}
              >
                Strict QC (10.50)
              </button>

              <button
                type="button"
                className="btn-pill-sample normal"
                onClick={() => applyPreset(FACTORY_DEFAULT_THRESHOLD, "Factory Calibrated")}
              >
                <Check size={12} />
                Factory Calibrated ({FACTORY_DEFAULT_THRESHOLD.toFixed(2)})
              </button>

              <button
                type="button"
                className="btn-pill-sample"
                onClick={() => applyPreset(14.5, "Permissive Tolerance")}
              >
                Permissive (14.50)
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Decision Engine Operating Principle */}
      <div className="card-panel">
        <div className="panel-header">
          <div className="panel-title">
            <Info size={18} color="#f59e0b" />
            <span>Decision Boundary Logic (PatchCore v2 Engine)</span>
          </div>
        </div>

        <div className="panel-body">
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              gap: "1rem",
            }}
          >
            <div
              style={{
                background: "var(--color-pass-bg)",
                border: "1px solid var(--color-pass-border)",
                borderRadius: "var(--radius-md)",
                padding: "1rem",
              }}
            >
              <div style={{ color: "var(--color-pass)", fontWeight: 700, fontSize: "0.95rem" }}>
                PASS Decision
              </div>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.35rem" }}>
                Score is <strong>below</strong> threshold minus 10% review margin (Score &lt;{" "}
                {(val * 0.9).toFixed(4)}). Cap passes downstream with high quality certainty.
              </p>
            </div>

            <div
              style={{
                background: "var(--color-review-bg)",
                border: "1px solid var(--color-review-border)",
                borderRadius: "var(--radius-md)",
                padding: "1rem",
              }}
            >
              <div style={{ color: "var(--color-review)", fontWeight: 700, fontSize: "0.95rem" }}>
                REVIEW Decision
              </div>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.35rem" }}>
                Score is <strong>within ±10% margin</strong> of threshold ({(val * 0.9).toFixed(4)} to{" "}
                {(val * 1.1).toFixed(4)}). Borderline item held for manual supervisor sign-off.
              </p>
            </div>

            <div
              style={{
                background: "var(--color-fail-bg)",
                border: "1px solid var(--color-fail-border)",
                borderRadius: "var(--radius-md)",
                padding: "1rem",
              }}
            >
              <div style={{ color: "var(--color-fail)", fontWeight: 700, fontSize: "0.95rem" }}>
                FAIL Decision
              </div>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.35rem" }}>
                Score is <strong>above</strong> threshold plus 10% review margin (Score &gt;{" "}
                {(val * 1.1).toFixed(4)}). Cap exhibits surface or contour defects and is rejected.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Hardware & System Specifications */}
      <div className="card-panel">
        <div className="panel-header">
          <div className="panel-title">
            <Cpu size={18} color="#a855f7" />
            <span>Inspection System Specifications</span>
          </div>
        </div>

        <div className="panel-body">
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
              gap: "1rem",
              fontSize: "0.85rem",
            }}
          >
            <div className="gate-card">
              <Cpu size={18} color="#38bdf8" />
              <div className="gate-info">
                <span className="gate-title">ML Architecture</span>
                <span className="gate-desc">PatchCore (Anomalib) + ROI 80% Crop</span>
              </div>
            </div>

            <div className="gate-card">
              <Database size={18} color="#10b981" />
              <div className="gate-info">
                <span className="gate-title">Data Storage</span>
                <span className="gate-desc">SQLite WAL + Local Artifact Storage</span>
              </div>
            </div>

            <div className="gate-card">
              <ShieldCheck size={18} color="#f59e0b" />
              <div className="gate-info">
                <span className="gate-title">Pre-Inference Gates</span>
                <span className="gate-desc">Laplacian Sharpness, Exposure, Detection</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
