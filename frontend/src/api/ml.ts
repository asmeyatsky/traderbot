import apiClient from './client';
import type { Prediction, Signal } from '../types/ml';

export async function getPrediction(symbol: string): Promise<Prediction> {
  const { data } = await apiClient.get<Prediction>(`/ml/predict/${symbol}`);
  return data;
}

export async function getSignal(symbol: string, userId: string): Promise<Signal> {
  const { data } = await apiClient.get<Signal>(`/ml/signal/${symbol}/${userId}`);
  return data;
}
