import React, { useEffect, useState } from "react";
import {
  Scan,
  BarChart3,
  Sliders,
  Volume2,
  VolumeX,
  Sun,
  Moon,
  Activity,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { api } from "../services/api";
import type { HealthStatus } from "../types/inspection";

export type ActiveTab = "inspection" | "analytics" | "settings";

interface HeaderProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  soundEnabled: boolean;
  onToggleSound: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onTabChange,
  soundEnabled,
  onToggleSound,
}) => {
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    try {
      return (localStorage.getItem("visionqc_theme") as "dark" | "light") || "dark";
    } catch {
      return "dark";
    }
  });

  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("visionqc_theme", theme);
    } catch {}
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  // Heartbeat health check
  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const res = await api.checkHealth();
        if (mounted) {
          setHealth(res);
          setIsHealthy(res.status === "ok");
        }
      } catch (e) {
        if (mounted) {
          setIsHealthy(false);
        }
      }
    };

    check();
    const interval = setInterval(check, 10000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="app-header">
      <div className="header-inner">
        {/* Brand & Station Info */}
        <div className="brand-section">
          <div className="logo-badge">
            <div className="logo-icon-wrap">
              <Scan size={20} />
            </div>
            <div>
              <div className="logo-text-title">
                VisionQC
                <span className="model-tag">PATCHCORE v2</span>
              </div>
            </div>
          </div>

          <div className="station-badge">
            <span className="station-dot" />
            <span>STATION 01 · LINE A</span>
          </div>
        </div>

        {/* Center Navigation Tabs */}
        <nav className="header-nav">
          <button
            type="button"
            className={`nav-tab ${activeTab === "inspection" ? "active" : ""}`}
            onClick={() => onTabChange("inspection")}
          >
            <Scan size={15} />
            Live Inspection
          </button>
          <button
            type="button"
            className={`nav-tab ${activeTab === "analytics" ? "active" : ""}`}
            onClick={() => onTabChange("analytics")}
          >
            <BarChart3 size={15} />
            Analytics & Audit
          </button>
          <button
            type="button"
            className={`nav-tab ${activeTab === "settings" ? "active" : ""}`}
            onClick={() => onTabChange("settings")}
          >
            <Sliders size={15} />
            Calibration
          </button>
        </nav>

        {/* Telemetry & Utility Controls */}
        <div className="header-actions">
          {/* Health Badge */}
          <div
            className={`telemetry-chip ${
              isHealthy === true ? "online" : isHealthy === false ? "offline" : ""
            }`}
            title="FastAPI Backend Health Status"
          >
            {isHealthy === true ? (
              <CheckCircle2 size={12} />
            ) : isHealthy === false ? (
              <AlertCircle size={12} />
            ) : (
              <Activity size={12} />
            )}
            <span>
              {isHealthy === true
                ? `Online (${health?.latencyMs}ms)`
                : isHealthy === false
                ? "Backend Offline"
                : "Connecting..."}
            </span>
          </div>

          {/* Sound Toggle */}
          <button
            type="button"
            className="icon-btn"
            onClick={onToggleSound}
            title={soundEnabled ? "Mute Inspection Sound Cues" : "Enable Inspection Sound Cues"}
          >
            {soundEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
          </button>

          {/* Theme Toggle */}
          <button
            type="button"
            className="icon-btn"
            onClick={toggleTheme}
            title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>
    </header>
  );
};
