export interface MarketData {
  symbol: string;
  current_price: number;
  price_change: number;
  price_change_percent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  previous_close: number;
  historical_data: HistoricalBar[];
  sentiment?: SentimentData;
  signals?: TradingSignal[];
  technical_indicators?: Record<string, number>;
}

export interface HistoricalBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface SentimentData {
  overall_sentiment: number;
  sentiment_label: string;
  news_count: number;
  articles: NewsArticle[];
}

export interface NewsArticle {
  title: string;
  source: string;
  published_at: string;
  sentiment_score: number;
  url: string;
}

export interface TradingSignal {
  indicator: string;
  signal: string;
  value: number;
  description: string;
}
