export interface Order {
  id: string;
  user_id: string;
  symbol: string;
  position_type: string;
  order_type: string;
  quantity: number;
  price: number | null;
  stop_price: number | null;
  status: string;
  filled_quantity: number;
  placed_at: string;
  executed_at: string | null;
  commission: number | null;
  notes: string | null;
}

/** Helper: maps position_type to user-friendly label */
export function sideLabel(positionType: string): string {
  return positionType === 'SHORT' ? 'SELL' : 'BUY';
}

export interface CreateOrderRequest {
  symbol: string;
  position_type: string;
  order_type: string;
  quantity: number;
  limit_price?: number;
  stop_price?: number;
  notes?: string;
}

export interface OrderListResponse {
  orders: Order[];
  total: number;
}
