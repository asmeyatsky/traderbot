import apiClient from './client';
import type { Portfolio, PortfolioAllocation } from '../types/portfolio';

export async function getPortfolio(): Promise<Portfolio> {
  const { data } = await apiClient.get<Portfolio>('/portfolio');
  return data;
}

export async function getPortfolioAllocation(): Promise<PortfolioAllocation> {
  const { data } = await apiClient.get<{
    cash_percentage: number;
    stocks_percentage: number;
    by_symbol: Record<string, number>;
    by_sector: Record<string, number>;
    timestamp: string;
  }>('/portfolio/allocation');
  // Transform backend by_symbol dict into AllocationItem array for the chart
  const allocations = Object.entries(data.by_symbol ?? {}).map(([symbol, percentage]) => ({
    symbol,
    percentage,
  }));
  return {
    allocations,
    cash_percentage: data.cash_percentage,
    stocks_percentage: data.stocks_percentage,
  };
}

export async function depositCash(amount: number): Promise<Portfolio> {
  const { data } = await apiClient.post<Portfolio>('/portfolio/cash-deposit', { amount });
  return data;
}

export async function withdrawCash(amount: number): Promise<Portfolio> {
  const { data } = await apiClient.post<Portfolio>('/portfolio/cash-withdraw', { amount });
  return data;
}
