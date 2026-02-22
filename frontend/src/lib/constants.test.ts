import { describe, it, expect } from 'vitest';
import { ORDER_SIDES, ORDER_TYPES, ORDER_STATUSES, RISK_TOLERANCES, INVESTMENT_GOALS } from './constants';

describe('constants', () => {
  it('defines order sides', () => {
    expect(ORDER_SIDES).toEqual(['BUY', 'SELL']);
  });

  it('defines order types', () => {
    expect(ORDER_TYPES).toContain('MARKET');
    expect(ORDER_TYPES).toContain('LIMIT');
    expect(ORDER_TYPES).toContain('STOP_LOSS');
    expect(ORDER_TYPES).toContain('TRAILING_STOP');
  });

  it('defines order statuses matching backend enum', () => {
    expect(ORDER_STATUSES).toContain('PENDING');
    expect(ORDER_STATUSES).toContain('EXECUTED');
    expect(ORDER_STATUSES).toContain('PARTIALLY_FILLED');
    expect(ORDER_STATUSES).toContain('CANCELLED');
    expect(ORDER_STATUSES).toContain('FAILED');
  });

  it('defines risk tolerances matching backend enum', () => {
    expect(RISK_TOLERANCES).toEqual(['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE']);
  });

  it('defines investment goals matching backend enum', () => {
    expect(INVESTMENT_GOALS).toEqual(['CAPITAL_PRESERVATION', 'BALANCED_GROWTH', 'MAXIMUM_RETURNS']);
  });
});
