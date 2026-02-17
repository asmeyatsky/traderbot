import apiClient from './client';
import type { RiskMetrics, StressResult } from '../types/risk';

export async function getPortfolioRisk(userId: string, lookbackDays = 252): Promise<RiskMetrics> {
  const { data } = await apiClient.get<RiskMetrics>(`/risk/portfolio/${userId}`, {
    params: { lookback_days: lookbackDays, confidence_level: 0.95 },
  });
  return data;
}

export async function runStressTest(userId: string, scenario: string): Promise<StressResult> {
  const { data } = await apiClient.post<StressResult>(`/risk/stress-test/${userId}`, null, {
    params: { scenario_name: scenario },
  });
  return data;
}
