export interface Market {
  market_code: string;
  market_name: string;
  country: string;
  currency: string;
}

export interface Stock {
  symbol: string;
  name: string;
  sector: string;
}

export interface MarketListResponse {
  markets: Market[];
}

export interface StockListResponse {
  market_code: string;
  stocks: Stock[];
}
