import apiClient from './client';
import type {
  AutoTradingSettings,
  UpdateAutoTradingRequest,
  ActivityLogResponse,
  ActivitySummary,
} from '../types/auto-trading';

export async function getAutoTradingSettings(): Promise<AutoTradingSettings> {
  const { data } = await apiClient.get<AutoTradingSettings>('/users/me/auto-trading');
  return data;
}

export async function updateAutoTradingSettings(
  payload: UpdateAutoTradingRequest,
): Promise<AutoTradingSettings> {
  const { data } = await apiClient.patch<AutoTradingSettings>('/users/me/auto-trading', payload);
  return data;
}

export async function getTradingActivity(
  skip: number,
  limit: number,
  eventType?: string,
): Promise<ActivityLogResponse> {
  const params: Record<string, string | number> = { skip, limit };
  if (eventType) params.event_type = eventType;
  const { data } = await apiClient.get<ActivityLogResponse>('/trading-activity', { params });
  return data;
}

export async function getTradingActivitySummary(): Promise<ActivitySummary> {
  const { data } = await apiClient.get<ActivitySummary>('/trading-activity/summary');
  return data;
}
