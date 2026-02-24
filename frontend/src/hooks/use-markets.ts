import { useQuery } from '@tanstack/react-query';
import * as marketsApi from '../api/markets';

export function useMarkets() {
  return useQuery({
    queryKey: ['markets'],
    queryFn: marketsApi.getMarkets,
  });
}

export function useStocks(marketCode: string | null, search: string) {
  return useQuery({
    queryKey: ['stocks', marketCode, search],
    queryFn: () => marketsApi.getStocks(marketCode!, search || undefined),
    enabled: !!marketCode,
  });
}
