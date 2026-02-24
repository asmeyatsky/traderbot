import { describe, it, expect, beforeEach } from 'vitest';
import apiClient from './client';
import { useAuthStore } from '../stores/auth-store';

describe('apiClient', () => {
  beforeEach(() => {
    useAuthStore.getState().logout();
  });

  it('has correct baseURL from env', () => {
    expect(apiClient.defaults.baseURL).toBe('/api/v1');
  });

  it('has JSON content type', () => {
    expect(apiClient.defaults.headers['Content-Type']).toBe('application/json');
  });

  it('adds auth header when token exists', () => {
    useAuthStore.getState().login('test-token', {
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
      allowed_markets: ['US_NYSE', 'US_NASDAQ'],
      created_at: '',
      updated_at: '',
    });

    const interceptor = apiClient.interceptors.request as unknown as { handlers: Array<{ fulfilled: (config: Record<string, unknown>) => Record<string, unknown> }> };
    const requestHandler = interceptor.handlers[0].fulfilled;
    const config = { headers: {} as Record<string, string> };
    const result = requestHandler(config);
    expect((result.headers as Record<string, string>).Authorization).toBe('Bearer test-token');
  });

  it('does not add auth header when no token', () => {
    const interceptor = apiClient.interceptors.request as unknown as { handlers: Array<{ fulfilled: (config: Record<string, unknown>) => Record<string, unknown> }> };
    const requestHandler = interceptor.handlers[0].fulfilled;
    const config = { headers: {} as Record<string, string> };
    const result = requestHandler(config);
    expect((result.headers as Record<string, string>).Authorization).toBeUndefined();
  });
});
