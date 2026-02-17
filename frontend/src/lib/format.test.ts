import { describe, it, expect } from 'vitest';
import { formatCurrency, formatPercent, formatDate, formatDateTime, formatNumber } from './format';

describe('formatCurrency', () => {
  it('formats positive amounts', () => {
    expect(formatCurrency(1234.56)).toBe('$1,234.56');
  });

  it('formats zero', () => {
    expect(formatCurrency(0)).toBe('$0.00');
  });

  it('formats negative amounts', () => {
    expect(formatCurrency(-500)).toBe('-$500.00');
  });

  it('rounds to 2 decimal places', () => {
    expect(formatCurrency(99.999)).toBe('$100.00');
  });

  it('formats large numbers with commas', () => {
    expect(formatCurrency(1000000)).toBe('$1,000,000.00');
  });
});

describe('formatPercent', () => {
  it('adds + for positive values', () => {
    expect(formatPercent(5.5)).toBe('+5.50%');
  });

  it('adds - for negative values', () => {
    expect(formatPercent(-3.2)).toBe('-3.20%');
  });

  it('adds + for zero', () => {
    expect(formatPercent(0)).toBe('+0.00%');
  });

  it('respects custom decimal places', () => {
    expect(formatPercent(1.2345, 1)).toBe('+1.2%');
  });
});

describe('formatDate', () => {
  it('formats ISO date strings', () => {
    const result = formatDate('2024-03-15T10:00:00Z');
    expect(result).toContain('Mar');
    expect(result).toContain('15');
    expect(result).toContain('2024');
  });

  it('formats Date objects', () => {
    const result = formatDate(new Date(2024, 0, 1));
    expect(result).toContain('Jan');
    expect(result).toContain('1');
    expect(result).toContain('2024');
  });
});

describe('formatDateTime', () => {
  it('includes time', () => {
    const result = formatDateTime('2024-06-15T14:30:00Z');
    expect(result).toContain('Jun');
    expect(result).toContain('15');
  });
});

describe('formatNumber', () => {
  it('formats with default 2 decimals', () => {
    expect(formatNumber(1234.5)).toBe('1,234.50');
  });

  it('formats with custom decimals', () => {
    expect(formatNumber(1234.5678, 0)).toBe('1,235');
  });
});
