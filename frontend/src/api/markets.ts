import apiClient from './client';
import type { MarketListResponse, StockListResponse } from '../types/market';

export async function getMarkets(): Promise<MarketListResponse> {
  const { data } = await apiClient.get<MarketListResponse>('/markets');
  return data;
}

export async function getStocks(marketCode: string, search?: string): Promise<StockListResponse> {
  const params: Record<string, string> = {};
  if (search) params.search = search;
  const { data } = await apiClient.get<StockListResponse>(`/markets/${marketCode}/stocks`, { params });
  return data;
}
