import { useQuery } from '@tanstack/react-query';
import * as riskApi from '../api/risk';

export function usePortfolioRisk(userId: string | undefined) {
  return useQuery({
    queryKey: ['risk', userId],
    queryFn: () => riskApi.getPortfolioRisk(userId!),
    enabled: !!userId,
  });
}
