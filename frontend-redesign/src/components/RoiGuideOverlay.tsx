import React from "react";

interface RoiGuideOverlayProps {
  isInspecting: boolean;
  isActive: boolean;
}

export const RoiGuideOverlay: React.FC<RoiGuideOverlayProps> = ({ isInspecting, isActive }) => {
  if (!isActive) return null;

  return (
    <>
      <svg
        className="roi-overlay"
        viewBox="0 0 400 300"
        preserveAspectRatio="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          {/* Radial mask to darken outer edges slightly to emphasize center 80% ROI */}
          <radialGradient id="roiGradient" cx="50%" cy="50%" r="50%">
            <stop offset="60%" stopColor="#000000" stopOpacity="0" />
            <stop offset="90%" stopColor="#000000" stopOpacity="0.45" />
          </radialGradient>
        </defs>

        {/* Ambient Darkening */}
        <rect width="400" height="300" fill="url(#roiGradient)" />

        {/* Outer Corner Alignment Markers */}
        <path d="M 20 50 L 20 20 L 50 20" className="reticle-corner" />
        <path d="M 350 20 L 380 20 L 380 50" className="reticle-corner" />
        <path d="M 20 250 L 20 280 L 50 280" className="reticle-corner" />
        <path d="M 350 280 L 380 280 L 380 250" className="reticle-corner" />

        {/* Center Crosshairs */}
        <line x1="200" y1="130" x2="200" y2="170" className="reticle-crosshair" />
        <line x1="180" y1="150" x2="220" y2="150" className="reticle-crosshair" />

        {/* 80% ROI Circular Inspection Target */}
        <ellipse
          cx="200"
          cy="150"
          rx="110"
          ry="105"
          className="reticle-circle"
          style={{
            stroke: isInspecting ? "#38bdf8" : "rgba(59, 130, 246, 0.7)",
            strokeWidth: isInspecting ? "2.5" : "2",
          }}
        />

        {/* Inner concentric fine target */}
        <ellipse
          cx="200"
          cy="150"
          rx="40"
          ry="38"
          stroke="rgba(255, 255, 255, 0.2)"
          strokeWidth="1"
          strokeDasharray="2 4"
          fill="none"
        />
      </svg>

      {/* Radar Sweep Animation while PatchCore is processing */}
      {isInspecting && <div className="scanner-sweep" />}

      {/* Helper label */}
      <div className="reticle-label">
        {isInspecting ? "⚡ RUNNING PATCHCORE INFERENCE..." : "ALIGN CAP IN 80% ROI TARGET"}
      </div>
    </>
  );
};
