import axios from 'axios';
import { API_BASE_URL } from '../lib/constants';
import { useAuthStore } from '../stores/auth-store';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

apiClient.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

let isRedirecting = false;

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      const url = error.config?.url ?? '';
      // Don't redirect on auth endpoints — let the form handle the error
      const isAuthEndpoint = url.includes('/users/login') || url.includes('/users/register') || url.includes('/users/logout');
      if (!isAuthEndpoint && !isRedirecting) {
        isRedirecting = true;
        useAuthStore.getState().logout();
        window.location.href = '/login';
        // Reset flag after navigation so future 401s (after re-login) still work
        setTimeout(() => { isRedirecting = false; }, 2000);
      }
    }
    return Promise.reject(error);
  },
);

export default apiClient;
