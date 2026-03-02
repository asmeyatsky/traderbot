import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import apiClient from '../api/client';
import { useAuthStore } from '../stores/auth-store';

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
      const user = useAuthStore.getState().user;
      // Return defaults from user data
      return {
        enabled: false,
        auto_trading_enabled: false,
        watchlist: [],
        trading_budget: null,
        stop_loss_pct: 5,
        take_profit_pct: 10,
        confidence_threshold: 0.6,
        max_position_pct: 20,
      } as AutoTradingSettings;
    },
  });
}

export function useUpdateAutoTrading() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (settings: Partial<AutoTradingSettings>) => {
      const { data } = await apiClient.patch('/users/me', settings);
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-trading-settings'] });
      queryClient.invalidateQueries({ queryKey: ['me'] });
    },
  });
}
