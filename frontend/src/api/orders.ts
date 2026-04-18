import { isAxiosError } from 'axios';
import apiClient from './client';
import type {
  CreateOrderRequest,
  DisciplineVetoErrorBody,
  Order,
  OrderListResponse,
} from '../types/order';

/** Parse a thrown error from `createOrder` and return the structured
 *  discipline-veto body when the server refused the trade with HTTP 400 +
 *  `{error: 'discipline_veto', vetoes: [...]}`. Returns null for any other
 *  error shape so callers can fall through to their generic error branch. */
export function extractDisciplineVeto(err: unknown): DisciplineVetoErrorBody | null {
  if (!isAxiosError(err)) return null;
  if (err.response?.status !== 400) return null;
  const detail = err.response.data?.detail;
  if (detail && typeof detail === 'object' && detail.error === 'discipline_veto') {
    return detail as DisciplineVetoErrorBody;
  }
  return null;
}

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
