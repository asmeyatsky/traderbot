import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import StatCard from './StatCard';

describe('StatCard', () => {
  it('renders title and value', () => {
    render(<StatCard title="Portfolio Value" value="$10,000.00" />);
    expect(screen.getByText('Portfolio Value')).toBeInTheDocument();
    expect(screen.getByText('$10,000.00')).toBeInTheDocument();
  });

  it('renders change text when provided', () => {
    render(<StatCard title="P&L" value="$500" change="+5.00%" changeType="positive" />);
    expect(screen.getByText('+5.00%')).toBeInTheDocument();
    expect(screen.getByText('+5.00%')).toHaveClass('text-green-600');
  });

  it('applies negative change color', () => {
    render(<StatCard title="P&L" value="$500" change="-3.00%" changeType="negative" />);
    expect(screen.getByText('-3.00%')).toHaveClass('text-red-600');
  });

  it('does not render change when not provided', () => {
    const { container } = render(<StatCard title="Count" value="42" />);
    const paragraphs = container.querySelectorAll('p');
    expect(paragraphs).toHaveLength(2); // title + value only
  });
});
