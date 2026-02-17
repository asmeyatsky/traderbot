import apiClient from './client';
import type { Portfolio, PortfolioAllocation } from '../types/portfolio';

export async function getPortfolio(): Promise<Portfolio> {
  const { data } = await apiClient.get<Portfolio>('/portfolio');
  return data;
}

export async function getPortfolioAllocation(): Promise<PortfolioAllocation> {
  const { data } = await apiClient.get<PortfolioAllocation>('/portfolio/allocation');
  return data;
}

export async function depositCash(amount: number): Promise<Portfolio> {
  const { data } = await apiClient.post<Portfolio>('/portfolio/cash-deposit', { amount });
  return data;
}

export async function withdrawCash(amount: number): Promise<Portfolio> {
  const { data } = await apiClient.post<Portfolio>('/portfolio/cash-withdraw', { amount });
  return data;
}
