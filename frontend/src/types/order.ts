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
  totp_code?: string;
  /** Phase 10.1 — set to true only after the user explicitly confirms they
   *  want to break one of their discipline rules. Every use is audited. */
  override_discipline_vetoes?: boolean;
}

/** Phase 10.1 — structured body the backend returns on HTTP 400 when an
 *  order violates a discipline rule. The UI presents these in a modal
 *  with an "Override and proceed" button that re-submits with the
 *  `override_discipline_vetoes` flag set. */
export interface DisciplineVeto {
  rule_id: string;
  rule_text: string;
  evidence: string;
}

export interface DisciplineVetoErrorBody {
  error: 'discipline_veto';
  message: string;
  vetoes: DisciplineVeto[];
}

export interface OrderListResponse {
  orders: Order[];
  total: number;
}
