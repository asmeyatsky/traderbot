import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { useAuthStore } from '../../stores/auth-store';
import ProtectedRoute from './ProtectedRoute';

function renderProtected(initialEntry = '/') {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/login" element={<div>Login Page</div>} />
        <Route element={<ProtectedRoute />}>
          <Route path="/" element={<div>Dashboard</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

describe('ProtectedRoute', () => {
  beforeEach(() => {
    useAuthStore.getState().logout();
  });

  it('redirects to /login when not authenticated', () => {
    renderProtected('/');
    expect(screen.getByText('Login Page')).toBeInTheDocument();
    expect(screen.queryByText('Dashboard')).not.toBeInTheDocument();
  });

  it('renders children when authenticated', () => {
    useAuthStore.getState().login('token', {
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
      sector_preferences: [],
      sector_exclusions: [],
      is_active: true,
      email_notifications_enabled: true,
      sms_notifications_enabled: false,
      approval_mode_enabled: false,
      created_at: '',
      updated_at: '',
    });
    renderProtected('/');
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.queryByText('Login Page')).not.toBeInTheDocument();
  });
});
