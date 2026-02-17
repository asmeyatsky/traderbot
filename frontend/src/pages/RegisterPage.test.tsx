import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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
  it('renders step 1 with account fields', () => {
    renderWithProviders(<RegisterPage />);
    expect(screen.getByText('Create your account')).toBeInTheDocument();
    expect(screen.getByLabelText('First Name')).toBeInTheDocument();
    expect(screen.getByLabelText('Last Name')).toBeInTheDocument();
    expect(screen.getByLabelText('Email')).toBeInTheDocument();
    expect(screen.getByLabelText('Password')).toBeInTheDocument();
  });

  it('advances to step 2 with risk profile cards', async () => {
    renderWithProviders(<RegisterPage />);
    await userEvent.type(screen.getByLabelText('First Name'), 'Test');
    await userEvent.type(screen.getByLabelText('Last Name'), 'User');
    await userEvent.type(screen.getByLabelText('Email'), 'test@example.com');
    await userEvent.type(screen.getByLabelText('Password'), 'password123');
    await userEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(screen.getByText('Select your risk tolerance')).toBeInTheDocument();
    expect(screen.getByText('Conservative')).toBeInTheDocument();
    expect(screen.getByText('Moderate')).toBeInTheDocument();
    expect(screen.getByText('Aggressive')).toBeInTheDocument();
  });

  it('advances to step 3 with investment goal cards', async () => {
    renderWithProviders(<RegisterPage />);
    await userEvent.type(screen.getByLabelText('First Name'), 'Test');
    await userEvent.type(screen.getByLabelText('Last Name'), 'User');
    await userEvent.type(screen.getByLabelText('Email'), 'test@example.com');
    await userEvent.type(screen.getByLabelText('Password'), 'password123');
    await userEvent.click(screen.getByRole('button', { name: 'Next' }));
    await userEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(screen.getByText('Choose your investment goal')).toBeInTheDocument();
    expect(screen.getByText('Capital Preservation')).toBeInTheDocument();
    expect(screen.getByText('Balanced Growth')).toBeInTheDocument();
    expect(screen.getByText('Maximum Returns')).toBeInTheDocument();
  });

  it('has link to login page', () => {
    renderWithProviders(<RegisterPage />);
    expect(screen.getByText('Sign in')).toHaveAttribute('href', '/login');
  });

  it('can go back from step 2', async () => {
    renderWithProviders(<RegisterPage />);
    await userEvent.type(screen.getByLabelText('First Name'), 'Test');
    await userEvent.type(screen.getByLabelText('Last Name'), 'User');
    await userEvent.type(screen.getByLabelText('Email'), 'test@example.com');
    await userEvent.type(screen.getByLabelText('Password'), 'password123');
    await userEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(screen.getByText('Select your risk tolerance')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Back' }));
    expect(screen.getByLabelText('First Name')).toBeInTheDocument();
  });

  it('disables Next when step 1 fields are empty', () => {
    renderWithProviders(<RegisterPage />);
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();
  });
});
