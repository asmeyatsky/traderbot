import apiClient from './client';
import type { LoginResponse, RegisterRequest, UpdateUserRequest, User } from '../types/user';

export async function login(email: string, password: string): Promise<LoginResponse> {
  const { data } = await apiClient.post<LoginResponse>('/users/login', { email, password });
  return data;
}

export async function register(payload: RegisterRequest): Promise<User> {
  const { data } = await apiClient.post<User>('/users/register', payload);
  return data;
}

export async function getMe(): Promise<User> {
  const { data } = await apiClient.get<User>('/users/me');
  return data;
}

export async function updateMe(payload: UpdateUserRequest): Promise<User> {
  const { data } = await apiClient.put<User>('/users/me', payload);
  return data;
}

export async function logout(): Promise<void> {
  await apiClient.post('/users/logout');
}
