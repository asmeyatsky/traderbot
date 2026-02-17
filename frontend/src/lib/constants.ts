export const API_BASE_URL = import.meta.env.VITE_API_URL ?? '/api/v1';

export const ORDER_SIDES = ['BUY', 'SELL'] as const;
export type OrderSide = (typeof ORDER_SIDES)[number];

export const ORDER_TYPES = ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'] as const;
export type OrderType = (typeof ORDER_TYPES)[number];

export const ORDER_STATUSES = ['PENDING', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'REJECTED'] as const;
export type OrderStatus = (typeof ORDER_STATUSES)[number];

export const RISK_TOLERANCES = ['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE'] as const;
export type RiskTolerance = (typeof RISK_TOLERANCES)[number];

export const INVESTMENT_GOALS = ['CAPITAL_PRESERVATION', 'BALANCED_GROWTH', 'MAXIMUM_RETURNS'] as const;
export type InvestmentGoal = (typeof INVESTMENT_GOALS)[number];
