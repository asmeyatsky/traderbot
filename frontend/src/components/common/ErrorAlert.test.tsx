import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ErrorAlert from './ErrorAlert';

describe('ErrorAlert', () => {
  it('displays the error message', () => {
    render(<ErrorAlert message="Something went wrong" />);
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('shows retry button when onRetry is provided', () => {
    render(<ErrorAlert message="Error" onRetry={() => {}} />);
    expect(screen.getByText('Try again')).toBeInTheDocument();
  });

  it('does not show retry button when onRetry is not provided', () => {
    render(<ErrorAlert message="Error" />);
    expect(screen.queryByText('Try again')).not.toBeInTheDocument();
  });

  it('calls onRetry when retry button is clicked', async () => {
    const onRetry = vi.fn();
    render(<ErrorAlert message="Error" onRetry={onRetry} />);
    await userEvent.click(screen.getByText('Try again'));
    expect(onRetry).toHaveBeenCalledOnce();
  });
});
