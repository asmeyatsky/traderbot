import { useQuery, useMutation } from '@tanstack/react-query';
import { runBacktest, listStrategies } from '../api/backtest';
import type { BacktestRequest, BacktestResponse } from '../types/backtest';

export function useStrategies() {
  return useQuery({
    queryKey: ['backtest-strategies'],
    queryFn: listStrategies,
    staleTime: 60_000 * 60,
  });
}

export function useRunBacktest() {
  return useMutation<BacktestResponse, Error, BacktestRequest>({
    mutationFn: runBacktest,
  });
}
