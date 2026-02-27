export interface Position {
  id: string;
  symbol: string;
  position_type: string;
  quantity: number;
  average_buy_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  pnl_percentage: number;
  day_change: number | null;
  day_change_percent: number | null;
  created_at: string;
  updated_at: string;
}

export interface Portfolio {
  user_id: string;
  cash_balance: number;
  total_value: number;
  positions: Position[];
  created_at: string;
  updated_at: string;
}

export interface AllocationItem {
  symbol: string;
  percentage: number;
}

export interface PortfolioAllocation {
  allocations: AllocationItem[];
  cash_percentage: number;
  stocks_percentage: number;
}
