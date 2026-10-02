import React from "react";

interface ScoreGaugeProps {
  score: number | null;
  threshold: number;
  reviewMargin?: number; // e.g. 0.10 (10%)
  confidence?: number | null;
}

export const ScoreGauge: React.FC<ScoreGaugeProps> = ({
  score,
  threshold,
  reviewMargin = 0.1,
  confidence,
}) => {
  // Setup visual scaling bounds
  // Usually threshold is ~12.38. We display a window from [minVal, maxVal]
  const windowMargin = Math.max(threshold * 0.4, 4);
  const minVal = Math.max(0, threshold - windowMargin);
  const maxVal = threshold + windowMargin;
  const range = maxVal - minVal;

  const clampPercent = (val: number) => {
    const pct = ((val - minVal) / range) * 100;
    return Math.max(2, Math.min(98, pct));
  };

  const thresholdPct = clampPercent(threshold);
  const reviewLowerVal = threshold - threshold * reviewMargin;
  const reviewUpperVal = threshold + threshold * reviewMargin;

  const reviewLowerPct = clampPercent(reviewLowerVal);
  const reviewUpperPct = clampPercent(reviewUpperVal);

  const scorePct = score !== null ? clampPercent(score) : null;
  const delta = score !== null ? score - threshold : null;

  return (
    <div className="score-gauge-wrap">
      <div className="gauge-header">
        <span className="gauge-header-title">PatchCore Anomaly Metric</span>
        <div className="gauge-scores-readout">
          <div className="score-item">
            <span
              className="score-item-val"
              style={{
                color:
                  score === null
                    ? "var(--text-muted)"
                    : score > reviewUpperVal
                    ? "var(--color-fail)"
                    : score < reviewLowerVal
                    ? "var(--color-pass)"
                    : "var(--color-review)",
              }}
            >
              {score !== null ? score.toFixed(4) : "—"}
            </span>
            <span className="score-item-lbl">Score</span>
          </div>

          <div className="score-item">
            <span className="score-item-val" style={{ color: "var(--text-main)" }}>
              {threshold.toFixed(4)}
            </span>
            <span className="score-item-lbl">Threshold</span>
          </div>

          <div className="score-item">
            <span
              className="score-item-val"
              style={{
                color:
                  delta === null
                    ? "var(--text-dim)"
                    : delta > 0
                    ? "var(--color-fail)"
                    : "var(--color-pass)",
              }}
            >
              {delta !== null ? (delta > 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4)) : "—"}
            </span>
            <span className="score-item-lbl">Delta (Score - Thresh)</span>
          </div>

          <div className="score-item">
            <span className="score-item-val" style={{ color: "#38bdf8" }}>
              {confidence != null ? `${confidence.toFixed(0)}%` : "—"}
            </span>
            <span className="score-item-lbl">Confidence</span>
          </div>
        </div>
      </div>

      {/* Visual Gauge Track */}
      <div className="gauge-track-container">
        {/* Background track with color gradients */}
        <div
          className="gauge-track-fill"
          style={{
            background: `linear-gradient(to right, 
              var(--color-pass) 0%, 
              var(--color-pass) ${reviewLowerPct}%, 
              var(--color-review) ${reviewLowerPct}%, 
              var(--color-review) ${reviewUpperPct}%, 
              var(--color-fail) ${reviewUpperPct}%, 
              var(--color-fail) 100%)`,
          }}
        />

        {/* Calibrated Threshold Marker */}
        <div
          className="threshold-marker-pin"
          style={{ left: `${thresholdPct}%` }}
          title={`Supervisor Threshold: ${threshold.toFixed(4)}`}
        >
          <div className="threshold-marker-tag">
            TH: {threshold.toFixed(2)}
          </div>
        </div>

        {/* Current Score Needle */}
        {scorePct !== null && (
          <div
            className="score-marker-needle"
            style={{ left: `${scorePct}%` }}
            title={`Anomaly Score: ${score?.toFixed(4)}`}
          >
            <div className="needle-pointer-top" />
            <div className="needle-line" />
            <div className="needle-pointer-bottom" />
          </div>
        )}
      </div>

      {/* Legend & Scale */}
      <div className="gauge-legend">
        <span>◀ PASS (Normal)</span>
        <span>REVIEW BAND (±{(reviewMargin * 100).toFixed(0)}%)</span>
        <span>FAIL (Defect) ▶</span>
      </div>
    </div>
  );
};
