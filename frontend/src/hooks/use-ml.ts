import { useQuery } from '@tanstack/react-query';
import apiClient from '../api/client';

export interface Prediction {
  symbol: string;
  signal: string;
  confidence: number;
  predicted_change: number;
  predicted_direction: string;
  current_price: number;
  predicted_price: number;
}

export function usePrediction(symbol: string) {
  return useQuery({
    queryKey: ['prediction', symbol],
    queryFn: async () => {
      const { data } = await apiClient.get(`/ml/predict/${symbol}`);
      return data as Prediction;
    },
    enabled: !!symbol,
    staleTime: 60_000,
  });
}
