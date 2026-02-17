export interface RiskMetrics {
  var_95: number;
  var_99: number;
  expected_shortfall: number;
  max_drawdown: number;
  volatility: number;
  beta: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  correlation_matrix: Record<string, Record<string, number>>;
  stress_results: StressResult[];
}

export interface StressResult {
  scenario: string;
  portfolio_impact: number;
  probability: number;
  description: string;
}
