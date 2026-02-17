import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import CashManagement from './CashManagement';

describe('CashManagement', () => {
  it('displays cash balance formatted', () => {
    render(<CashManagement cashBalance={5000} totalValue={20000} />);
    expect(screen.getByText('$5,000.00')).toBeInTheDocument();
  });

  it('shows correct percentage', () => {
    render(<CashManagement cashBalance={5000} totalValue={20000} />);
    expect(screen.getByText('25.0% of portfolio')).toBeInTheDocument();
  });

  it('handles zero total value', () => {
    render(<CashManagement cashBalance={0} totalValue={0} />);
    expect(screen.getByText('0.0% of portfolio')).toBeInTheDocument();
  });
});
