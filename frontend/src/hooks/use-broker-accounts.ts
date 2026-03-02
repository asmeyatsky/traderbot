import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import apiClient from '../api/client';

export interface BrokerAccount {
  id: string;
  broker_type: string;
  paper_trading: boolean;
  label: string | null;
  is_active: boolean;
  api_key_hint: string;
  created_at: string;
  updated_at: string;
}

export function useBrokerAccounts() {
  return useQuery({
    queryKey: ['broker-accounts'],
    queryFn: async () => {
      const { data } = await apiClient.get('/broker-accounts');
      return data as BrokerAccount[];
    },
  });
}

export function useLinkBrokerAccount() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      broker_type: string;
      api_key: string;
      secret_key: string;
      paper_trading: boolean;
      label?: string;
    }) => {
      const { data } = await apiClient.post('/broker-accounts', body);
      return data as BrokerAccount;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['broker-accounts'] });
    },
  });
}

export function useUpdateBrokerAccount() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({
      id,
      ...body
    }: {
      id: string;
      paper_trading?: boolean;
      is_active?: boolean;
    }) => {
      const { data } = await apiClient.patch(`/broker-accounts/${id}`, body);
      return data as BrokerAccount;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['broker-accounts'] });
    },
  });
}

export function useDeleteBrokerAccount() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (id: string) => {
      await apiClient.delete(`/broker-accounts/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['broker-accounts'] });
    },
  });
}
