import { describe, it, expect, beforeEach } from 'vitest';
import { useAuthStore } from './auth-store';
import type { User } from '../types/user';

const mockUser: User = {
  id: 'user-1',
  email: 'test@example.com',
  first_name: 'John',
  last_name: 'Doe',
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
  allowed_markets: ['US_NYSE', 'US_NASDAQ'],
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
};

describe('auth-store', () => {
  beforeEach(() => {
    useAuthStore.getState().logout();
  });

  it('starts unauthenticated', () => {
    const state = useAuthStore.getState();
    expect(state.isAuthenticated).toBe(false);
    expect(state.token).toBeNull();
    expect(state.user).toBeNull();
  });

  it('login sets token, user and isAuthenticated', () => {
    useAuthStore.getState().login('jwt-token-123', mockUser);
    const state = useAuthStore.getState();
    expect(state.isAuthenticated).toBe(true);
    expect(state.token).toBe('jwt-token-123');
    expect(state.user).toEqual(mockUser);
  });

  it('logout clears state', () => {
    useAuthStore.getState().login('jwt-token-123', mockUser);
    useAuthStore.getState().logout();
    const state = useAuthStore.getState();
    expect(state.isAuthenticated).toBe(false);
    expect(state.token).toBeNull();
    expect(state.user).toBeNull();
  });

  it('setUser updates user without affecting token', () => {
    useAuthStore.getState().login('jwt-token-123', mockUser);
    const updatedUser = { ...mockUser, first_name: 'Jane' };
    useAuthStore.getState().setUser(updatedUser);
    const state = useAuthStore.getState();
    expect(state.user?.first_name).toBe('Jane');
    expect(state.token).toBe('jwt-token-123');
    expect(state.isAuthenticated).toBe(true);
  });
});
