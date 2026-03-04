import apiClient from './client';
import type { BacktestRequest, BacktestResponse, StrategyInfo } from '../types/backtest';

export async function runBacktest(params: BacktestRequest): Promise<BacktestResponse> {
  const { data } = await apiClient.post<BacktestResponse>('/backtest/run', params);
  return data;
}

export async function listStrategies(): Promise<{ strategies: StrategyInfo[] }> {
  const { data } = await apiClient.get<{ strategies: StrategyInfo[] }>('/backtest/strategies');
  return data;
}
