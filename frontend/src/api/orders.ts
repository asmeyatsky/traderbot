import apiClient from './client';
import type { CreateOrderRequest, Order, OrderListResponse } from '../types/order';

export async function createOrder(payload: CreateOrderRequest): Promise<Order> {
  const { data } = await apiClient.post<Order>('/orders/create', payload);
  return data;
}

export async function getOrders(skip = 0, limit = 50, statusFilter?: string): Promise<OrderListResponse> {
  const params: Record<string, string | number> = { skip, limit };
  if (statusFilter) params.status_filter = statusFilter;
  const { data } = await apiClient.get<OrderListResponse>('/orders', { params });
  return data;
}

export async function getOrder(orderId: string): Promise<Order> {
  const { data } = await apiClient.get<Order>(`/orders/${orderId}`);
  return data;
}

export async function cancelOrder(orderId: string): Promise<void> {
  await apiClient.delete(`/orders/${orderId}`);
}
