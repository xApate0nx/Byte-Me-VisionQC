import React, { useState } from "react";
import { Eye, Layers, Columns, Image as ImageIcon, Download, Maximize2 } from "lucide-react";

interface HeatmapViewerProps {
  imageUrl: string;
  heatmapUrl: string | null;
  altText?: string;
}

type ViewMode = "split" | "overlay" | "heatmap" | "original";

export const HeatmapViewer: React.FC<HeatmapViewerProps> = ({
  imageUrl,
  heatmapUrl,
  altText = "Inspection Artifact",
}) => {
  const [mode, setMode] = useState<ViewMode>("split");
  const [overlayOpacity, setOverlayOpacity] = useState<number>(65);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  // If there's no heatmap (e.g. invalid capture), fallback to original
  const hasHeatmap = Boolean(heatmapUrl);
  const activeMode = !hasHeatmap ? "original" : mode;

  return (
    <div className="visualizer-wrap">
      <div className="visualizer-nav">
        {hasHeatmap ? (
          <div className="view-mode-tabs">
            <button
              type="button"
              className={`view-mode-tab ${activeMode === "split" ? "active" : ""}`}
              onClick={() => setMode("split")}
            >
              <Columns size={13} style={{ display: "inline", marginRight: 4, verticalAlign: "middle" }} />
              Side-by-Side
            </button>
            <button
              type="button"
              className={`view-mode-tab ${activeMode === "overlay" ? "active" : ""}`}
              onClick={() => setMode("overlay")}
            >
              <Layers size={13} style={{ display: "inline", marginRight: 4, verticalAlign: "middle" }} />
              Overlay
            </button>
            <button
              type="button"
              className={`view-mode-tab ${activeMode === "heatmap" ? "active" : ""}`}
              onClick={() => setMode("heatmap")}
            >
              <Eye size={13} style={{ display: "inline", marginRight: 4, verticalAlign: "middle" }} />
              Heatmap
            </button>
            <button
              type="button"
              className={`view-mode-tab ${activeMode === "original" ? "active" : ""}`}
              onClick={() => setMode("original")}
            >
              <ImageIcon size={13} style={{ display: "inline", marginRight: 4, verticalAlign: "middle" }} />
              Original
            </button>
          </div>
        ) : (
          <span style={{ fontSize: "0.75rem", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
            Original Camera Frame (No Heatmap Generated)
          </span>
        )}

        <div style={{ display: "flex", gap: "0.4rem" }}>
          <button
            type="button"
            className="icon-btn"
            style={{ width: 28, height: 28 }}
            title="Download Inspection Image"
            onClick={() => {
              const link = document.createElement("a");
              link.href = heatmapUrl || imageUrl;
              link.download = "visionqc-inspection.jpg";
              link.click();
            }}
          >
            <Download size={14} />
          </button>
          <button
            type="button"
            className="icon-btn"
            style={{ width: 28, height: 28 }}
            title={isFullscreen ? "Exit Fullscreen" : "Expand Fullscreen"}
            onClick={() => setIsFullscreen(!isFullscreen)}
          >
            <Maximize2 size={14} />
          </button>
        </div>
      </div>

      {/* Main Display Area */}
      <div
        className="visualizer-display"
        style={isFullscreen ? { maxHeight: "70vh", height: "70vh" } : undefined}
      >
        {activeMode === "split" && hasHeatmap && (
          <div className="split-view-container">
            <div className="split-view-pane">
              <img src={imageUrl} alt="Original Capture" className="visualizer-img" />
              <div className="split-view-label">ORIGINAL CAPTURE</div>
            </div>
            <div className="split-view-pane">
              <img src={heatmapUrl!} alt={altText} className="visualizer-img" />
              <div className="split-view-label">PATCHCORE HEATMAP</div>
            </div>
          </div>
        )}

        {activeMode === "overlay" && hasHeatmap && (
          <div style={{ position: "relative", width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <img src={imageUrl} alt="Original Cap" className="visualizer-img" />
            <img
              src={heatmapUrl!}
              alt="Heatmap Overlay"
              className="visualizer-img visualizer-overlay-layer"
              style={{
                opacity: overlayOpacity / 100,
                mixBlendMode: "screen",
              }}
            />
          </div>
        )}

        {activeMode === "heatmap" && hasHeatmap && (
          <img src={heatmapUrl!} alt={altText} className="visualizer-img" />
        )}

        {activeMode === "original" && (
          <img src={imageUrl} alt="Original Inspection" className="visualizer-img" />
        )}
      </div>

      {/* Interactive Overlay Opacity Control */}
      {activeMode === "overlay" && hasHeatmap && (
        <div className="overlay-slider-control">
          <span>Original (0%)</span>
          <input
            type="range"
            min="0"
            max="100"
            value={overlayOpacity}
            onChange={(e) => setOverlayOpacity(Number(e.target.value))}
          />
          <span>Heatmap ({overlayOpacity}%)</span>
        </div>
      )}
    </div>
  );
};
