import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import PortfolioSummaryCard from './PortfolioSummaryCard';
import type { DashboardOverview } from '../../types/dashboard';

const mockData: DashboardOverview = {
  portfolio_value: 50000,
  daily_pnl: 250,
  daily_pnl_percent: 0.5,
  total_pnl: 5000,
  total_pnl_percent: 11.11,
  positions_count: 8,
  top_performers: [],
  worst_performers: [],
  allocation: [],
  performance_history: [],
};

describe('PortfolioSummaryCard', () => {
  it('renders portfolio value', () => {
    render(<PortfolioSummaryCard data={mockData} />);
    expect(screen.getByText('$50,000.00')).toBeInTheDocument();
  });

  it('renders daily P&L', () => {
    render(<PortfolioSummaryCard data={mockData} />);
    expect(screen.getByText('$250.00')).toBeInTheDocument();
  });

  it('renders total P&L', () => {
    render(<PortfolioSummaryCard data={mockData} />);
    expect(screen.getByText('$5,000.00')).toBeInTheDocument();
  });

  it('renders positions count', () => {
    render(<PortfolioSummaryCard data={mockData} />);
    expect(screen.getByText('8')).toBeInTheDocument();
  });

  it('renders all stat card titles', () => {
    render(<PortfolioSummaryCard data={mockData} />);
    expect(screen.getByText('Portfolio Value')).toBeInTheDocument();
    expect(screen.getByText('Daily P&L')).toBeInTheDocument();
    expect(screen.getByText('Total P&L')).toBeInTheDocument();
    expect(screen.getByText('Positions')).toBeInTheDocument();
  });
});
