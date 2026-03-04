export interface BacktestRequest {
  strategy: string;
  symbol: string;
  start_date?: string;
  end_date?: string;
  initial_capital?: number;
}

export interface BacktestTrade {
  date: string;
  symbol: string;
  action: string;
  quantity: number;
  price: number;
  commission: number;
  reason: string;
  value: number;
}

export interface BacktestResponse {
  strategy: string;
  symbol: string;
  initial_capital: number;
  final_value: number;
  total_return_pct: number;
  annualized_return_pct: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  trades: BacktestTrade[];
  error?: string;
}

export interface StrategyInfo {
  name: string;
  label: string;
  description: string;
}
