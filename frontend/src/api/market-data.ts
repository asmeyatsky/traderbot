import apiClient from './client';
import type { MarketData } from '../types/market-data';

export async function getEnhancedMarketData(symbol: string, days = 30): Promise<MarketData> {
  const { data } = await apiClient.get<MarketData>(`/market-data/enhanced/${symbol}`, {
    params: { include_news: true, include_technical: true, days },
  });
  return data;
}
