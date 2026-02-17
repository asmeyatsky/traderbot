import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SymbolSearch from './SymbolSearch';

describe('SymbolSearch', () => {
  it('renders input and search button', () => {
    render(<SymbolSearch onSearch={() => {}} />);
    expect(screen.getByPlaceholderText(/search symbol/i)).toBeInTheDocument();
    expect(screen.getByText('Search')).toBeInTheDocument();
  });

  it('calls onSearch with uppercased symbol on submit', async () => {
    const onSearch = vi.fn();
    render(<SymbolSearch onSearch={onSearch} />);
    const input = screen.getByPlaceholderText(/search symbol/i);
    await userEvent.type(input, 'aapl');
    await userEvent.click(screen.getByText('Search'));
    expect(onSearch).toHaveBeenCalledWith('AAPL');
  });

  it('does not call onSearch with empty input', async () => {
    const onSearch = vi.fn();
    render(<SymbolSearch onSearch={onSearch} />);
    await userEvent.click(screen.getByText('Search'));
    expect(onSearch).not.toHaveBeenCalled();
  });

  it('shows current symbol in input', () => {
    render(<SymbolSearch onSearch={() => {}} currentSymbol="MSFT" />);
    expect(screen.getByDisplayValue('MSFT')).toBeInTheDocument();
  });
});
