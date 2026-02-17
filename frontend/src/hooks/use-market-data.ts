import { useQuery } from '@tanstack/react-query';
import * as marketDataApi from '../api/market-data';

export function useEnhancedMarketData(symbol: string) {
  return useQuery({
    queryKey: ['market-data', symbol],
    queryFn: () => marketDataApi.getEnhancedMarketData(symbol),
    enabled: !!symbol,
    staleTime: 60_000,
  });
}
