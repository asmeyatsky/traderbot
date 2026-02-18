export interface AutoTradingSettings {
  enabled: boolean;
  watchlist: string[];
  trading_budget: number | null;
}

export interface UpdateAutoTradingRequest {
  enabled?: boolean;
  watchlist?: string[];
  trading_budget?: number | null;
}

export interface ActivityLogItem {
  id: string;
  event_type: string;
  symbol: string | null;
  signal: string | null;
  confidence: number | null;
  quantity: number | null;
  price: number | null;
  message: string;
  created_at: string;
}

export interface ActivityLogResponse {
  items: ActivityLogItem[];
  skip: number;
  limit: number;
  count: number;
}

export interface ActivitySummary {
  summary: Record<string, number>;
}
