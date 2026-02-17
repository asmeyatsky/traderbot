export interface Position {
  symbol: string;
  quantity: number;
  average_cost: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
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
  value: number;
  percentage: number;
  sector?: string;
}

export interface PortfolioAllocation {
  allocations: AllocationItem[];
  total_value: number;
  cash_percentage: number;
}
