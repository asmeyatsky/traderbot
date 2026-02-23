export const API_BASE_URL = import.meta.env.VITE_API_URL ?? '/api/v1';

export const ORDER_SIDES = ['BUY', 'SELL'] as const;
export type OrderSide = (typeof ORDER_SIDES)[number];

export const ORDER_TYPES = ['MARKET', 'LIMIT', 'STOP_LOSS', 'TRAILING_STOP'] as const;
export type OrderType = (typeof ORDER_TYPES)[number];

export const ORDER_STATUSES = ['PENDING', 'EXECUTED', 'PARTIALLY_FILLED', 'CANCELLED', 'FAILED'] as const;
export type OrderStatus = (typeof ORDER_STATUSES)[number];

export const RISK_TOLERANCES = ['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE'] as const;
export type RiskTolerance = (typeof RISK_TOLERANCES)[number];

export const INVESTMENT_GOALS = ['CAPITAL_PRESERVATION', 'BALANCED_GROWTH', 'MAXIMUM_RETURNS'] as const;
export type InvestmentGoal = (typeof INVESTMENT_GOALS)[number];

export type RiskPresetKey = 'CONSERVATIVE' | 'MODERATE' | 'AGGRESSIVE' | 'CUSTOM';

export interface RiskPresetValues {
  stopLoss: number;
  takeProfit: number;
  confidence: number;
  maxPosition: number;
}

export const RISK_PRESETS: Record<Exclude<RiskPresetKey, 'CUSTOM'>, RiskPresetValues> = {
  CONSERVATIVE: { stopLoss: 3, takeProfit: 8, confidence: 80, maxPosition: 10 },
  MODERATE: { stopLoss: 5, takeProfit: 15, confidence: 65, maxPosition: 20 },
  AGGRESSIVE: { stopLoss: 10, takeProfit: 30, confidence: 55, maxPosition: 35 },
};
