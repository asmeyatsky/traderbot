import { useQuery } from '@tanstack/react-query';
import * as mlApi from '../api/ml';

export function usePrediction(symbol: string) {
  return useQuery({
    queryKey: ['prediction', symbol],
    queryFn: () => mlApi.getPrediction(symbol),
    enabled: !!symbol,
  });
}

export function useSignal(symbol: string, userId: string | undefined) {
  return useQuery({
    queryKey: ['signal', symbol, userId],
    queryFn: () => mlApi.getSignal(symbol, userId!),
    enabled: !!symbol && !!userId,
  });
}
