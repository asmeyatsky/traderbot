import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import apiClient from '../api/client';

export interface AutoTradingSettings {
  enabled: boolean;
  auto_trading_enabled: boolean;
  watchlist: string[];
  trading_budget: number | null;
  stop_loss_pct: number;
  take_profit_pct: number;
  confidence_threshold: number;
  max_position_pct: number;
}

export function useAutoTradingSettings() {
  return useQuery({
    queryKey: ['auto-trading-settings'],
    queryFn: async () => {
      const { data } = await apiClient.get('/users/me/auto-trading');
      return data as AutoTradingSettings;
    },
  });
}

export function useUpdateAutoTrading() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (settings: Partial<AutoTradingSettings>) => {
      const { data } = await apiClient.patch('/users/me/auto-trading', settings);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-trading-settings'] });
      queryClient.invalidateQueries({ queryKey: ['me'] });
    },
  });
}
