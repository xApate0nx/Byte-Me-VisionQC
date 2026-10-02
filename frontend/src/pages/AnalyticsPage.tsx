import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  AlertCircle,
  RefreshCw,
  Download,
  Search,
  ExternalLink,
  Layers,
  TrendingDown,
  ShieldCheck,
} from "lucide-react";
import { api } from "../services/api";
import type { Decision, InspectionResult, TodayStats } from "../types/inspection";
import { InspectionDetailModal } from "../components/InspectionDetailModal";

interface AnalyticsPageProps {
  refreshTrigger: number;
  showToast: (msg: string, type: "success" | "error" | "warning" | "info") => void;
}

const emptyStats: TodayStats = {
  total: 0,
  passed: 0,
  failed: 0,
  review: 0,
  invalid: 0,
  rejection_rate: 0,
};

export const AnalyticsPage: React.FC<AnalyticsPageProps> = ({ refreshTrigger, showToast }) => {
  const [stats, setStats] = useState<TodayStats>(emptyStats);
  const [history, setHistory] = useState<InspectionResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedInspection, setSelectedInspection] = useState<InspectionResult | null>(null);
  const [filterResult, setFilterResult] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const [nextStats, nextHistory] = await Promise.all([
        api.getStats(),
        api.getHistory(100),
      ]);
      setStats(nextStats);
      setHistory(nextHistory.inspections);
    } catch (e) {
      console.error("Failed to load analytics data:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData, refreshTrigger]);

  // Auto refresh interval if enabled
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchData]);

  // Filtered Inspections list
  const filteredInspections = useMemo(() => {
    return history.filter((item) => {
      // Filter by result status
      if (filterResult !== "ALL" && item.result !== filterResult) {
        return false;
      }
      // Filter by query (ID, reason, or filename)
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase();
        const matchesId = String(item.inspection_id).includes(query);
        const matchesReason = item.reason?.toLowerCase().includes(query);
        const matchesFilename = item.filename?.toLowerCase().includes(query);
        if (!matchesId && !matchesReason && !matchesFilename) return false;
      }
      return true;
    });
  }, [history, filterResult, searchQuery]);

  // CSV Export
  const handleExportCSV = () => {
    if (!history.length) {
      showToast("No inspection records available to export.", "info");
      return;
    }

    const headers = [
      "ID",
      "Timestamp",
      "Result",
      "Anomaly Score",
      "Threshold",
      "Confidence (%)",
      "Quality Valid",
      "Product Detected",
      "Alignment Valid",
      "Reason",
      "Filename",
    ];

    const rows = history.map((item) => [
      item.inspection_id ?? item.id ?? "",
      item.timestamp,
      item.result,
      item.anomaly_score !== null ? item.anomaly_score.toFixed(6) : "",
      item.threshold,
      item.confidence != null ? item.confidence.toFixed(1) : "",
      item.image_quality_valid ?? "",
      item.product_detected ?? "",
      item.alignment_valid ?? "",
      `"${(item.reason || "").replace(/"/g, '""')}"`,
      item.filename || "",
    ]);

    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `visionqc-inspections-${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
    showToast("Exported inspection audit records as CSV.", "success");
  };

  const passRate = stats.total > 0 ? ((stats.passed / stats.total) * 100).toFixed(1) : "0.0";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Top Metric KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-top">
            <span className="kpi-title">Total Inspected</span>
            <div className="kpi-icon">
              <Layers size={18} />
            </div>
          </div>
          <div className="kpi-val">{stats.total}</div>
          <div className="kpi-sub">
            <span>Water cap units evaluated today</span>
          </div>
        </div>

        <div className="kpi-card" style={{ borderLeft: "4px solid var(--color-pass)" }}>
          <div className="kpi-top">
            <span className="kpi-title">Pass Rate / Yield</span>
            <div className="kpi-icon" style={{ color: "var(--color-pass)" }}>
              <ShieldCheck size={18} />
            </div>
          </div>
          <div className="kpi-val" style={{ color: "var(--color-pass)" }}>
            {passRate}%
          </div>
          <div className="kpi-sub">
            <span>{stats.passed} compliant caps passed</span>
          </div>
        </div>

        <div className="kpi-card" style={{ borderLeft: "4px solid var(--color-fail)" }}>
          <div className="kpi-top">
            <span className="kpi-title">Rejection Rate</span>
            <div className="kpi-icon" style={{ color: "var(--color-fail)" }}>
              <TrendingDown size={18} />
            </div>
          </div>
          <div className="kpi-val" style={{ color: "var(--color-fail)" }}>
            {stats.rejection_rate.toFixed(1)}%
          </div>
          <div className="kpi-sub">
            <span>{stats.failed} defective caps rejected</span>
          </div>
        </div>

        <div className="kpi-card" style={{ borderLeft: "4px solid var(--color-review)" }}>
          <div className="kpi-top">
            <span className="kpi-title">Review / Invalid</span>
            <div className="kpi-icon" style={{ color: "var(--color-review)" }}>
              <AlertTriangle size={18} />
            </div>
          </div>
          <div className="kpi-val" style={{ color: "var(--color-review)" }}>
            {stats.review + stats.invalid}
          </div>
          <div className="kpi-sub">
            <span>
              {stats.review} review · {stats.invalid} invalid
            </span>
          </div>
        </div>
      </div>

      {/* Historical Audit Table */}
      <div className="table-panel">
        <div className="table-toolbar">
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
            {/* Filter Pills */}
            <div className="table-filter-pills">
              {["ALL", "PASS", "FAIL", "REVIEW", "INSPECTION_INVALID"].map((tab) => (
                <button
                  key={tab}
                  type="button"
                  className={`filter-pill ${filterResult === tab ? "active" : ""}`}
                  onClick={() => setFilterResult(tab)}
                >
                  {tab === "ALL"
                    ? `All (${history.length})`
                    : tab === "INSPECTION_INVALID"
                    ? "Invalid"
                    : tab}
                </button>
              ))}
            </div>

            {/* Search Input */}
            <div style={{ position: "relative" }}>
              <Search
                size={14}
                style={{
                  position: "absolute",
                  left: 10,
                  top: "50%",
                  transform: "translateY(-50%)",
                  color: "var(--text-dim)",
                }}
              />
              <input
                type="text"
                placeholder="Search ID, reason, filename..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="table-search-input"
                style={{ paddingLeft: 30 }}
              />
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            {/* Auto refresh checkbox */}
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.35rem",
                fontSize: "0.75rem",
                color: "var(--text-muted)",
                cursor: "pointer",
                marginRight: 6,
              }}
            >
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              Auto-poll (5s)
            </label>

            {/* Refresh Button */}
            <button
              type="button"
              className="btn-secondary"
              onClick={fetchData}
              disabled={loading}
              title="Refresh log"
            >
              <RefreshCw size={14} className={loading ? "spin" : ""} />
              Refresh
            </button>

            {/* Export CSV Button */}
            <button
              type="button"
              className="btn-secondary"
              onClick={handleExportCSV}
              title="Download CSV audit log"
            >
              <Download size={14} />
              Export CSV
            </button>
          </div>
        </div>

        {/* Inspections Table */}
        <div className="table-wrapper">
          <table className="qc-table">
            <thead>
              <tr>
                <th style={{ width: 60 }}>Thumbnail</th>
                <th>ID</th>
                <th>Time (UTC)</th>
                <th>Result</th>
                <th>Anomaly Score</th>
                <th>Threshold</th>
                <th>Confidence</th>
                <th>Reason</th>
                <th style={{ textAlign: "right" }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredInspections.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ textAlign: "center", padding: "2.5rem 1rem", color: "var(--text-dim)" }}>
                    No inspection records found matching your filters.
                  </td>
                </tr>
              ) : (
                filteredInspections.map((item, index) => {
                  const itemId = item.inspection_id ?? item.id ?? index + 1;
                  return (
                    <tr
                      key={itemId}
                      onClick={() => setSelectedInspection(item)}
                      style={{ cursor: "pointer" }}
                    >
                      <td>
                        <div
                          style={{
                            width: 40,
                            height: 40,
                            borderRadius: "var(--radius-sm)",
                            background: "#020617",
                            overflow: "hidden",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            border: "1px solid var(--border-subtle)",
                          }}
                        >
                          {item.heatmap_url || item.image_url ? (
                            <img
                              src={item.heatmap_url || item.image_url}
                              alt=""
                              style={{ width: "100%", height: "100%", objectFit: "cover" }}
                            />
                          ) : (
                            <Layers size={14} color="var(--text-dim)" />
                          )}
                        </div>
                      </td>
                      <td className="mono" style={{ fontWeight: 600 }}>
                        #{itemId}
                      </td>
                      <td className="mono" style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
                        {new Date(item.timestamp).toLocaleTimeString()}
                      </td>
                      <td>
                        <span className={`badge-decision ${item.result.toLowerCase()}`}>
                          {item.result === "PASS" && <CheckCircle2 size={12} />}
                          {item.result === "FAIL" && <XCircle size={12} />}
                          {item.result === "REVIEW" && <AlertTriangle size={12} />}
                          {item.result === "INSPECTION_INVALID" && <AlertCircle size={12} />}
                          {item.result}
                        </span>
                      </td>
                      <td className="mono" style={{ fontWeight: 600 }}>
                        {item.anomaly_score !== null ? item.anomaly_score.toFixed(4) : "—"}
                      </td>
                      <td className="mono" style={{ color: "var(--text-muted)" }}>
                        {item.threshold.toFixed(4)}
                      </td>
                      <td className="mono" style={{ color: "#38bdf8" }}>
                        {item.confidence != null ? `${item.confidence.toFixed(0)}%` : "—"}
                      </td>
                      <td
                        style={{
                          maxWidth: 220,
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          color: "var(--text-muted)",
                          fontSize: "0.8rem",
                        }}
                        title={item.reason || ""}
                      >
                        {item.reason || "—"}
                      </td>
                      <td style={{ textAlign: "right" }}>
                        <button
                          type="button"
                          className="btn-secondary"
                          style={{ padding: "0.3rem 0.65rem", fontSize: "0.75rem" }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedInspection(item);
                          }}
                        >
                          <ExternalLink size={12} />
                          Details
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Deep-Dive Inspection Modal */}
      <InspectionDetailModal
        inspection={selectedInspection}
        onClose={() => setSelectedInspection(null)}
      />
    </div>
  );
};
