import apiClient from './client';
import type { ScreenResponse, PrebuiltScreenInfo } from '../types/screening';

export async function screenStocks(params: Record<string, unknown>): Promise<ScreenResponse> {
  const { data } = await apiClient.post<ScreenResponse>('/screening/screen', params);
  return data;
}

export async function listPrebuiltScreens(): Promise<{ screens: PrebuiltScreenInfo[] }> {
  const { data } = await apiClient.get<{ screens: PrebuiltScreenInfo[] }>('/screening/prebuilt');
  return data;
}
