import apiClient from './client';
import type { DashboardOverview, TechnicalIndicators } from '../types/dashboard';

export async function getDashboardOverview(userId: string, days = 30): Promise<DashboardOverview> {
  const { data } = await apiClient.get<DashboardOverview>(`/dashboard/overview/${userId}`, {
    params: { include_technical: true, days },
  });
  return data;
}

export async function getTechnicalIndicators(symbol: string, days = 30): Promise<TechnicalIndicators> {
  const { data } = await apiClient.get<TechnicalIndicators>(`/dashboard/technical-indicators/${symbol}`, {
    params: { days },
  });
  return data;
}
