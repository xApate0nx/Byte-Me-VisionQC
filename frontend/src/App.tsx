import { useState, useEffect, useCallback } from "react";
import { Header, type ActiveTab } from "./components/Header";
import { LiveInspectionPage } from "./pages/LiveInspectionPage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { SettingsPage } from "./pages/SettingsPage";
import { Toast, type ToastMessage } from "./components/Toast";
import { useSoundFeedback } from "./hooks/useSoundFeedback";
import { api } from "./services/api";

const DEFAULT_THRESHOLD = 12.3785223961;

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("inspection");
  const [threshold, setThreshold] = useState<number>(DEFAULT_THRESHOLD);
  const [refreshTrigger, setRefreshTrigger] = useState<number>(0);
  const [toast, setToast] = useState<ToastMessage | null>(null);

  const sound = useSoundFeedback();

  const showToast = useCallback(
    (message: string, type: "success" | "error" | "warning" | "info" = "info") => {
      setToast({
        id: String(Date.now()),
        type,
        message,
      });
    },
    []
  );

  // Load initial threshold from FastAPI backend
  useEffect(() => {
    api
      .getThreshold()
      .then((res) => {
        if (res.threshold) {
          setThreshold(res.threshold);
        }
      })
      .catch((err) => {
        console.warn("Could not retrieve threshold from backend:", err);
      });
  }, []);

  // Quick keyboard shortcuts for tab navigation (1, 2, 3)
  useEffect(() => {
    const handleKeyNav = (e: KeyboardEvent) => {
      // Don't trigger if user is typing in an input
      if (["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }
      if (e.key === "1") setActiveTab("inspection");
      if (e.key === "2") setActiveTab("analytics");
      if (e.key === "3") setActiveTab("settings");
    };

    window.addEventListener("keydown", handleKeyNav);
    return () => window.removeEventListener("keydown", handleKeyNav);
  }, []);

  const handleInspectionFinished = () => {
    setRefreshTrigger((prev) => prev + 1);
  };

  const handleThresholdUpdated = (newVal: number) => {
    setThreshold(newVal);
    setRefreshTrigger((prev) => prev + 1);
  };

  return (
    <div className="app-container">
      {/* Global Application Header */}
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        soundEnabled={sound.enabled}
        onToggleSound={sound.toggleSound}
      />

      {/* Main Dynamic View */}
      <main className="main-content">
        {activeTab === "inspection" && (
          <LiveInspectionPage
            threshold={threshold}
            onInspectionFinished={handleInspectionFinished}
            soundCuePass={sound.playPass}
            soundCueFail={sound.playFail}
            soundCueReview={sound.playReview}
            showToast={showToast}
          />
        )}

        {activeTab === "analytics" && (
          <AnalyticsPage refreshTrigger={refreshTrigger} showToast={showToast} />
        )}

        {activeTab === "settings" && (
          <SettingsPage
            currentThreshold={threshold}
            onThresholdUpdated={handleThresholdUpdated}
            showToast={showToast}
          />
        )}
      </main>

      {/* Floating Notifications */}
      <Toast toast={toast} onDismiss={() => setToast(null)} />
    </div>
  );
}
