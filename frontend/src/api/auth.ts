import apiClient from './client';
import type { LoginResponse, RegisterRequest, UpdateUserRequest, User } from '../types/user';

export async function login(username: string, password: string): Promise<LoginResponse> {
  const params = new URLSearchParams();
  params.append('username', username);
  params.append('password', password);
  const { data } = await apiClient.post<LoginResponse>('/users/login', params, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  });
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
