import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import * as ordersApi from '../api/orders';
import type { CreateOrderRequest } from '../types/order';

export function useOrders(statusFilter?: string) {
  return useQuery({
    queryKey: ['orders', statusFilter],
    queryFn: () => ordersApi.getOrders(0, 100, statusFilter),
  });
}

export function useCreateOrder() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: CreateOrderRequest) => ordersApi.createOrder(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
      queryClient.invalidateQueries({ queryKey: ['portfolio'] });
    },
  });
}

export function useCancelOrder() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (orderId: string) => ordersApi.cancelOrder(orderId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
    },
  });
}
