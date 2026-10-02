export type Decision = "PASS" | "REVIEW" | "FAIL" | "INSPECTION_INVALID";

export interface InspectionResult {
  inspection_id?: number;
  id?: number;
  filename: string | null;
  anomaly_score: number | null;
  threshold: number;
  result: Decision;
  confidence?: number | null;
  reason?: string | null;
  image_quality_valid?: boolean | number | null;
  quality_reason?: string | null;
  product_detected?: boolean | number | null;
  alignment_valid?: boolean | number | null;
  image_url: string;
  heatmap_url: string | null;
  timestamp: string;
}

export interface TodayStats {
  total: number;
  passed: number;
  failed: number;
  review: number;
  invalid: number;
  rejection_rate: number;
}

export interface HealthStatus {
  status: string;
  latencyMs: number;
  timestamp: string;
}

export interface ThresholdSetting {
  threshold: number;
}
