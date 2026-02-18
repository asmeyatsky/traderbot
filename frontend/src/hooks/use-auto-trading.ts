import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as api from '../api/auto-trading';
import type { UpdateAutoTradingRequest } from '../types/auto-trading';

export function useAutoTradingSettings() {
  return useQuery({
    queryKey: ['auto-trading'],
    queryFn: api.getAutoTradingSettings,
  });
}

export function useUpdateAutoTrading() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: UpdateAutoTradingRequest) => api.updateAutoTradingSettings(payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['auto-trading'] }),
  });
}

export function useTradingActivity(skip: number, limit: number, eventType?: string) {
  return useQuery({
    queryKey: ['trading-activity', skip, limit, eventType],
    queryFn: () => api.getTradingActivity(skip, limit, eventType),
  });
}

export function useTradingActivitySummary() {
  return useQuery({
    queryKey: ['trading-activity', 'summary'],
    queryFn: api.getTradingActivitySummary,
  });
}
