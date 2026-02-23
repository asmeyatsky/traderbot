export interface AutoTradingSettings {
  enabled: boolean;
  watchlist: string[];
  trading_budget: number | null;
  stop_loss_pct: number;
  take_profit_pct: number;
  confidence_threshold: number;
  max_position_pct: number;
}

export interface UpdateAutoTradingRequest {
  enabled?: boolean;
  watchlist?: string[];
  trading_budget?: number | null;
  stop_loss_pct?: number;
  take_profit_pct?: number;
  confidence_threshold?: number;
  max_position_pct?: number;
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
