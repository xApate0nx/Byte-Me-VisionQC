import type { HealthStatus, InspectionResult, TodayStats } from "../types/inspection";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const message = body?.detail || `Server responded with status ${response.status}`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  async inspect(file: Blob, filename = "capture.jpg"): Promise<InspectionResult> {
    const data = new FormData();
    data.append("file", file, filename);
    return request<InspectionResult>("/api/inspect", {
      method: "POST",
      body: data,
    });
  },

  async getThreshold(): Promise<{ threshold: number }> {
    return request<{ threshold: number }>("/api/settings/threshold");
  },

  async updateThreshold(threshold: number): Promise<{ threshold: number }> {
    return request<{ threshold: number }>("/api/settings/threshold", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold }),
    });
  },

  async getStats(): Promise<TodayStats> {
    return request<TodayStats>("/api/stats/today");
  },

  async getHistory(limit = 100): Promise<{ inspections: InspectionResult[] }> {
    return request<{ inspections: InspectionResult[] }>(`/api/inspections?limit=${limit}`);
  },

  async checkHealth(): Promise<HealthStatus> {
    const start = performance.now();
    const data = await request<{ status: string }>("/api/health");
    const latency = Math.round(performance.now() - start);
    return {
      status: data.status,
      latencyMs: latency,
      timestamp: new Date().toISOString(),
    };
  },

  async loadSampleImage(sampleUrl: string, sampleName: string): Promise<File> {
    const response = await fetch(sampleUrl);
    if (!response.ok) {
      throw new Error(`Failed to load sample image from ${sampleUrl}`);
    }
    const blob = await response.blob();
    return new File([blob], sampleName, { type: blob.type || "image/jpeg" });
  },
};
