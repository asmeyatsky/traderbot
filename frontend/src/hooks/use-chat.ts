import { useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useChatStore } from '../stores/chat-store';
import * as chatApi from '../api/chat';
import type { ChatMessage, TradeAction } from '../types/chat';

export function useConversations() {
  const setConversations = useChatStore((s) => s.setConversations);

  return useQuery({
    queryKey: ['conversations'],
    queryFn: async () => {
      const data = await chatApi.listConversations();
      setConversations(data.conversations);
      return data;
    },
  });
}

export function useConversation(id: string | null) {
  return useQuery({
    queryKey: ['conversation', id],
    queryFn: () => chatApi.getConversation(id!),
    enabled: !!id,
  });
}

export function useCreateConversation() {
  const queryClient = useQueryClient();
  const addConversation = useChatStore((s) => s.addConversation);
  const setActiveConversation = useChatStore((s) => s.setActiveConversation);

  return useMutation({
    mutationFn: chatApi.createConversation,
    onSuccess: (conv) => {
      addConversation(conv);
      setActiveConversation(conv.id);
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
    },
  });
}

export function useDeleteConversation() {
  const queryClient = useQueryClient();
  const removeConversation = useChatStore((s) => s.removeConversation);

  return useMutation({
    mutationFn: chatApi.deleteConversation,
    onSuccess: (_, id) => {
      removeConversation(id);
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
    },
  });
}

export function useSendMessage() {
  const addPendingMessage = useChatStore((s) => s.addPendingMessage);
  const clearPendingMessages = useChatStore((s) => s.clearPendingMessages);
  const appendStreamingContent = useChatStore((s) => s.appendStreamingContent);
  const setIsStreaming = useChatStore((s) => s.setIsStreaming);
  const clearStreaming = useChatStore((s) => s.clearStreaming);
  const queryClient = useQueryClient();

  const sendMessage = useCallback(
    async (conversationId: string, content: string) => {
      // Add optimistic user message
      const userMsg: ChatMessage = {
        id: `temp-${Date.now()}`,
        conversation_id: conversationId,
        role: 'user',
        content,
        created_at: new Date().toISOString(),
        trade_actions: [],
      };
      addPendingMessage(conversationId, userMsg);

      // Start streaming
      setIsStreaming(true);
      clearStreaming();

      let fullContent = '';

      try {
        for await (const event of chatApi.sendMessageStream(
          conversationId,
          content,
        )) {
          switch (event.type) {
            case 'text_delta':
              fullContent += event.content ?? '';
              appendStreamingContent(event.content ?? '');
              break;

            case 'tool_result':
              // Tool results are processed by the AI, no direct UI update needed
              break;

            case 'done': {
              // Add the complete assistant message
              const metadata = event.metadata as Record<string, unknown> | undefined;
              const actions = (metadata?.trade_actions ?? []) as TradeAction[];
              const assistantMsg: ChatMessage = {
                id: (metadata?.message_id as string) ?? `ai-${Date.now()}`,
                conversation_id: conversationId,
                role: 'assistant',
                content: fullContent,
                created_at: new Date().toISOString(),
                trade_actions: actions,
              };
              addPendingMessage(conversationId, assistantMsg);
              clearStreaming();
              break;
            }

            case 'error': {
              clearStreaming();
              const errorMsg: ChatMessage = {
                id: `err-${Date.now()}`,
                conversation_id: conversationId,
                role: 'assistant',
                content: event.content ?? 'An error occurred.',
                created_at: new Date().toISOString(),
                trade_actions: [],
              };
              addPendingMessage(conversationId, errorMsg);
              break;
            }
          }
        }
      } catch {
        clearStreaming();
        const errorMsg: ChatMessage = {
          id: `err-${Date.now()}`,
          conversation_id: conversationId,
          role: 'assistant',
          content: 'Failed to get a response. Please try again.',
          created_at: new Date().toISOString(),
          trade_actions: [],
        };
        addPendingMessage(conversationId, errorMsg);
      }

      // Refresh conversations list and specific conversation
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      queryClient.invalidateQueries({ queryKey: ['conversation', conversationId] });

      // Clear pending once server data is refetched
      // Small delay to let React Query refetch before clearing
      setTimeout(() => clearPendingMessages(conversationId), 500);
    },
    [addPendingMessage, clearPendingMessages, appendStreamingContent, setIsStreaming, clearStreaming, queryClient],
  );

  return { sendMessage };
}
