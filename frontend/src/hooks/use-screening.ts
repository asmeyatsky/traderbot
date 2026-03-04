import { useQuery, useMutation } from '@tanstack/react-query';
import { screenStocks, listPrebuiltScreens } from '../api/screening';
import type { ScreenResponse } from '../types/screening';

export function usePrebuiltScreens() {
  return useQuery({
    queryKey: ['prebuilt-screens'],
    queryFn: listPrebuiltScreens,
    staleTime: 60_000 * 60, // 1 hour — static data
  });
}

export function useScreenStocks() {
  return useMutation<ScreenResponse, Error, Record<string, unknown>>({
    mutationFn: screenStocks,
  });
}
