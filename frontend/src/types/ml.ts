export interface Prediction {
  symbol: string;
  predicted_direction: string;
  confidence: number;
  predicted_price: number;
  current_price: number;
  score: number;
  explanation: string;
}

export interface Signal {
  symbol: string;
  signal: string;
  confidence: number;
  news_impact: string;
  risk_adjusted: boolean;
}
