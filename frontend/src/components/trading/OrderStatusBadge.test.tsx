import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import OrderStatusBadge from './OrderStatusBadge';

describe('OrderStatusBadge', () => {
  it('renders PENDING with yellow styling', () => {
    render(<OrderStatusBadge status="PENDING" />);
    const badge = screen.getByText('PENDING');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass('bg-yellow-100', 'text-yellow-800');
  });

  it('renders FILLED with green styling', () => {
    render(<OrderStatusBadge status="FILLED" />);
    const badge = screen.getByText('FILLED');
    expect(badge).toHaveClass('bg-green-100', 'text-green-800');
  });

  it('renders CANCELLED with gray styling', () => {
    render(<OrderStatusBadge status="CANCELLED" />);
    const badge = screen.getByText('CANCELLED');
    expect(badge).toHaveClass('bg-gray-100', 'text-gray-800');
  });

  it('renders REJECTED with red styling', () => {
    render(<OrderStatusBadge status="REJECTED" />);
    const badge = screen.getByText('REJECTED');
    expect(badge).toHaveClass('bg-red-100', 'text-red-800');
  });

  it('handles unknown status gracefully', () => {
    render(<OrderStatusBadge status="UNKNOWN" />);
    expect(screen.getByText('UNKNOWN')).toBeInTheDocument();
  });
});
