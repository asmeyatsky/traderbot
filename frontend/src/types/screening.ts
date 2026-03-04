export interface ScreenResult {
  symbol: string;
  name: string;
  price: number;
  change_pct: number;
  volume: number;
  market_cap: number | null;
  sector: string | null;
  rsi: number | null;
}

export interface ScreenResponse {
  screen: string;
  count: number;
  results: ScreenResult[];
}

export interface PrebuiltScreenInfo {
  name: string;
  label: string;
  description: string;
}
