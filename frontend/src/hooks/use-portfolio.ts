import { useQuery } from '@tanstack/react-query';
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
