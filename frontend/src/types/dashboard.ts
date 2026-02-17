export interface DashboardOverview {
  portfolio_value: number;
  daily_pnl: number;
  daily_pnl_percent: number;
  total_pnl: number;
  total_pnl_percent: number;
  positions_count: number;
  top_performers: PerformanceItem[];
  worst_performers: PerformanceItem[];
  allocation: AllocationBreakdown[];
  performance_history: PerformancePoint[];
  technical_indicators?: Record<string, TechnicalIndicators>;
}

export interface PerformanceItem {
  symbol: string;
  change_percent: number;
  current_price: number;
}

export interface AllocationBreakdown {
  name: string;
  value: number;
  percentage: number;
}

export interface PerformancePoint {
  date: string;
  value: number;
}

export interface TechnicalIndicators {
  sma_20: number;
  sma_50: number;
  ema_12: number;
  ema_26: number;
  rsi: number;
  macd: number;
  macd_signal: number;
  macd_histogram: number;
  bollinger_upper: number;
  bollinger_middle: number;
  bollinger_lower: number;
  atr: number;
}
