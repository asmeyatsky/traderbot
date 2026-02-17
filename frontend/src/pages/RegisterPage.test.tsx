import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders } from '../test/test-utils';
import RegisterPage from './RegisterPage';

vi.mock('../hooks/use-auth', () => ({
  useRegister: () => ({
    mutate: vi.fn(),
    isPending: false,
    error: null,
  }),
  useLogin: () => ({
    mutate: vi.fn(),
    isPending: false,
    error: null,
  }),
}));

describe('RegisterPage', () => {
  it('renders registration form with all required fields', () => {
    renderWithProviders(<RegisterPage />);
    expect(screen.getByText('Create your account')).toBeInTheDocument();
    expect(screen.getByLabelText('First Name')).toBeInTheDocument();
    expect(screen.getByLabelText('Last Name')).toBeInTheDocument();
    expect(screen.getByLabelText('Email')).toBeInTheDocument();
    expect(screen.getByLabelText('Password')).toBeInTheDocument();
    expect(screen.getByLabelText('Risk Tolerance')).toBeInTheDocument();
    expect(screen.getByLabelText('Investment Goal')).toBeInTheDocument();
  });

  it('has risk tolerance options matching backend enum', () => {
    renderWithProviders(<RegisterPage />);
    const select = screen.getByLabelText('Risk Tolerance');
    expect(select).toBeInTheDocument();
    expect(screen.getByText('CONSERVATIVE')).toBeInTheDocument();
    expect(screen.getByText('MODERATE')).toBeInTheDocument();
    expect(screen.getByText('AGGRESSIVE')).toBeInTheDocument();
  });

  it('has investment goal options matching backend enum', () => {
    renderWithProviders(<RegisterPage />);
    expect(screen.getByText('CAPITAL PRESERVATION')).toBeInTheDocument();
    expect(screen.getByText('BALANCED GROWTH')).toBeInTheDocument();
    expect(screen.getByText('MAXIMUM RETURNS')).toBeInTheDocument();
  });

  it('has link to login page', () => {
    renderWithProviders(<RegisterPage />);
    expect(screen.getByText('Sign in')).toHaveAttribute('href', '/login');
  });

  it('defaults risk tolerance to MODERATE', () => {
    renderWithProviders(<RegisterPage />);
    const select = screen.getByLabelText('Risk Tolerance') as HTMLSelectElement;
    expect(select.value).toBe('MODERATE');
  });

  it('defaults investment goal to BALANCED_GROWTH', () => {
    renderWithProviders(<RegisterPage />);
    const select = screen.getByLabelText('Investment Goal') as HTMLSelectElement;
    expect(select.value).toBe('BALANCED_GROWTH');
  });
});
