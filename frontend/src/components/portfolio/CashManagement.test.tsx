import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CashManagement from './CashManagement';

function renderWithClient(ui: React.ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
}

describe('CashManagement', () => {
  it('displays cash balance formatted', () => {
    renderWithClient(<CashManagement cashBalance={5000} totalValue={20000} />);
    expect(screen.getByText('$5,000.00')).toBeInTheDocument();
  });

  it('shows correct percentage', () => {
    renderWithClient(<CashManagement cashBalance={5000} totalValue={20000} />);
    expect(screen.getByText('25.0% of portfolio')).toBeInTheDocument();
  });

  it('handles zero total value', () => {
    renderWithClient(<CashManagement cashBalance={0} totalValue={0} />);
    expect(screen.getByText('0.0% of portfolio')).toBeInTheDocument();
  });
});
