import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '../api/client';

export interface Strategy {
  id: string;
  user_id: string;
  author_name: string;
  name: string;
  description: string;
  strategy_type: string;
  parameters: Record<string, unknown>;
  symbol: string;
  is_public: boolean;
  fork_count: number;
  follower_count: number;
  best_return_pct: number | null;
  best_sharpe: number | null;
  created_at: string;
  is_following: boolean;
}

export interface LeaderboardEntry {
  rank: number;
  strategy_id: string;
  strategy_name: string;
  author_name: string;
  strategy_type: string;
  symbol: string;
  total_return_pct: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  win_rate: number;
  follower_count: number;
  is_following: boolean;
}

export function useMyStrategies() {
  return useQuery<Strategy[]>({
    queryKey: ['strategies', 'mine'],
    queryFn: async () => {
      const { data } = await api.get('/api/v1/strategies');
      return data;
    },
  });
}

export function useMarketplace() {
  return useQuery<Strategy[]>({
    queryKey: ['strategies', 'marketplace'],
    queryFn: async () => {
      const { data } = await api.get('/api/v1/strategies/marketplace');
      return data;
    },
  });
}

export function useLeaderboard() {
  return useQuery<LeaderboardEntry[]>({
    queryKey: ['strategies', 'leaderboard'],
    queryFn: async () => {
      const { data } = await api.get('/api/v1/strategies/leaderboard');
      return data;
    },
  });
}

export function useCreateStrategy() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      name: string;
      description?: string;
      strategy_type: string;
      parameters?: Record<string, unknown>;
      symbol?: string;
      is_public?: boolean;
    }) => {
      const { data } = await api.post('/api/v1/strategies', body);
      return data as Strategy;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['strategies'] });
    },
  });
}

export function useUpdateStrategy() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, ...body }: { id: string; name?: string; description?: string; is_public?: boolean; parameters?: Record<string, unknown>; symbol?: string }) => {
      const { data } = await api.patch(`/api/v1/strategies/${id}`, body);
      return data as Strategy;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['strategies'] });
    },
  });
}

export function useDeleteStrategy() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (id: string) => {
      await api.delete(`/api/v1/strategies/${id}`);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['strategies'] });
    },
  });
}

export function useSaveBacktestResult() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      strategy_id: string;
      symbol: string;
      initial_capital: number;
      final_value: number;
      total_return_pct: number;
      sharpe_ratio: number;
      max_drawdown_pct: number;
      win_rate: number;
      total_trades: number;
      volatility: number;
      profit_factor: number;
    }) => {
      const { data } = await api.post(`/api/v1/strategies/${body.strategy_id}/backtest`, body);
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['strategies'] });
    },
  });
}

export function useFollowStrategy() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (strategyId: string) => {
      const { data } = await api.post(`/api/v1/strategies/${strategyId}/follow`);
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['strategies'] });
    },
  });
}

export function useUnfollowStrategy() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (strategyId: string) => {
      const { data } = await api.delete(`/api/v1/strategies/${strategyId}/follow`);
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['strategies'] });
    },
  });
}

export function useForkStrategy() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (strategyId: string) => {
      const { data } = await api.post(`/api/v1/strategies/${strategyId}/fork`);
      return data as Strategy;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['strategies'] });
    },
  });
}
