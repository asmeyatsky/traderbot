import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as portfolioApi from '../api/portfolio';

export function usePortfolio() {
  return useQuery({
    queryKey: ['portfolio'],
    queryFn: portfolioApi.getPortfolio,
  });
}

export function usePortfolioAllocation() {
  return useQuery({
    queryKey: ['portfolio', 'allocation'],
    queryFn: portfolioApi.getPortfolioAllocation,
  });
}

export function useDepositCash() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: portfolioApi.depositCash,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolio'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard'] });
    },
  });
}

export function useWithdrawCash() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: portfolioApi.withdrawCash,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolio'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard'] });
    },
  });
}
