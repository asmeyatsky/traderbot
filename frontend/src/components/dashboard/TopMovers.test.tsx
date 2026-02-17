import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import TopMovers from './TopMovers';

describe('TopMovers', () => {
  it('renders gainers and losers sections', () => {
    render(<TopMovers gainers={[]} losers={[]} />);
    expect(screen.getByText('Top Gainers')).toBeInTheDocument();
    expect(screen.getByText('Top Losers')).toBeInTheDocument();
  });

  it('shows no data message when empty', () => {
    render(<TopMovers gainers={[]} losers={[]} />);
    expect(screen.getAllByText('No data')).toHaveLength(2);
  });

  it('renders gainer items', () => {
    render(
      <TopMovers
        gainers={[{ symbol: 'AAPL', change_percent: 5.2, current_price: 175 }]}
        losers={[]}
      />,
    );
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText('$175.00')).toBeInTheDocument();
    expect(screen.getByText('+5.20%')).toBeInTheDocument();
  });

  it('renders loser items', () => {
    render(
      <TopMovers
        gainers={[]}
        losers={[{ symbol: 'TSLA', change_percent: -3.1, current_price: 200 }]}
      />,
    );
    expect(screen.getByText('TSLA')).toBeInTheDocument();
    expect(screen.getByText('-3.10%')).toBeInTheDocument();
  });
});
