import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { useAuthStore } from '../../stores/auth-store';
import { useOnboardingStore } from '../../stores/onboarding-store';
import ProtectedRoute from './ProtectedRoute';

function renderProtected(initialEntry = '/dashboard') {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/login" element={<div>Login Page</div>} />
        <Route path="/onboarding" element={<div>Onboarding Page</div>} />
        <Route element={<ProtectedRoute />}>
          <Route path="/dashboard" element={<div>Dashboard</div>} />
          <Route path="/onboarding" element={<div>Onboarding</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

const testUser = {
  id: '1',
  email: 'test@test.com',
  first_name: 'Test',
  last_name: 'User',
  risk_tolerance: 'MODERATE',
  investment_goal: 'BALANCED_GROWTH',
  max_position_size_percentage: 10,
  daily_loss_limit: null,
  weekly_loss_limit: null,
  monthly_loss_limit: null,
  sector_preferences: [] as string[],
  sector_exclusions: [] as string[],
  is_active: true,
  email_notifications_enabled: true,
  sms_notifications_enabled: false,
  approval_mode_enabled: false,
  created_at: '',
  updated_at: '',
};

describe('ProtectedRoute', () => {
  beforeEach(() => {
    useAuthStore.getState().logout();
    useOnboardingStore.getState().reset();
  });

  it('redirects to /login when not authenticated', () => {
    renderProtected('/dashboard');
    expect(screen.getByText('Login Page')).toBeInTheDocument();
    expect(screen.queryByText('Dashboard')).not.toBeInTheDocument();
  });

  it('redirects to /onboarding when onboarding not completed', () => {
    useAuthStore.getState().login('token', testUser);
    renderProtected('/dashboard');
    expect(screen.getByText('Onboarding Page')).toBeInTheDocument();
    expect(screen.queryByText('Dashboard')).not.toBeInTheDocument();
  });

  it('renders children when authenticated and onboarding completed', () => {
    useAuthStore.getState().login('token', testUser);
    useOnboardingStore.getState().markComplete();
    renderProtected('/dashboard');
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.queryByText('Login Page')).not.toBeInTheDocument();
  });
});
