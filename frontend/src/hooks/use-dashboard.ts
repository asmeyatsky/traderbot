import { useQuery } from '@tanstack/react-query';
import * as dashboardApi from '../api/dashboard';

export function useDashboardOverview(userId: string | undefined) {
  return useQuery({
    queryKey: ['dashboard', 'overview', userId],
    queryFn: () => dashboardApi.getDashboardOverview(userId!),
    enabled: !!userId,
  });
}

export function useTechnicalIndicators(symbol: string) {
  return useQuery({
    queryKey: ['technical-indicators', symbol],
    queryFn: () => dashboardApi.getTechnicalIndicators(symbol),
    enabled: !!symbol,
  });
}
